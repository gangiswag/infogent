import os
import json
import re
from langchain_community.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.agents import Tool
from autogpt import AutoGPT
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_experimental.autonomous_agents.autogpt.output_parser import preprocess_json_input
from tqdm import tqdm
import faiss
from urllib.parse import unquote
from bs4 import BeautifulSoup
import markdownify
import requests
from typing import List, Dict
from cachetools import TTLCache, cached
import tiktoken
import functools
import argparse
from copy import deepcopy
import time
import datetime
import httpx
import pandas as pd

class BFSAgent:
    def __init__(self, args) -> None:

        self.NAVIGATOR_MODEL=args.navigator_model
        self.AGGREGATOR_MODEL=args.aggregator_model
        self.EXTRACTOR_MODEL=args.extractor_model
        self.search_tool = GoogleSerperAPIWrapper()
        self.enc = tiktoken.get_encoding("cl100k_base")

        self.aggregator_messages = list()

        self.num_iterations = args.num_iterations
        self.num_to_aggregate = args.num_to_aggregate
        self.timeout = 10

        self.wikipedia = httpx.Client(
            base_url="https://en.wikipedia.org/w/api.php",
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30,
        )
        self.wiki_errors = list()

        self.aggregated_data = list()
        self.counter = 1
        self.aggregator_cost = 0.0
        self.extractor_cost = 0.0
        self.extractor_time = 0.0
        self.aggregator_time = 0.0
        self.search_time = 0.0
        self.parse_time = 0.0

        self.current_query = None
        self.last_search_thought = "Not Available"
        self.last_selection_thought = "Not Available"
        self.previous_iterations = list()

        self.init_nav_agent()

    def setup(self, user_task):
        self.user_task = user_task

    def init_nav_agent(self):

        embeddings_model = OpenAIEmbeddings() 
        tools = [
                    Tool(
                        name = "search",
                        func=self.search,
                        description="Useful for when you need to gather information from the web. You should ask targeted questions"
                    ),
                    Tool(
                        name = "extract",
                        func=self.extract,
                        description="Useful to extract relevant information from provided URL and get feedback from Aggregator Module on how to proceed next"
                    )
                ]
        
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        if "claude" in self.NAVIGATOR_MODEL:
            agent_llm = ChatAnthropic(temperature=0.7, model=self.NAVIGATOR_MODEL)
        elif "gemini" in self.NAVIGATOR_MODEL:
            agent_llm = ChatGoogleGenerativeAI(temperature=0.7, model=self.NAVIGATOR_MODEL, convert_system_message_to_human=True)
        else:
            agent_llm = ChatOpenAI(temperature=0.7, model_name=self.NAVIGATOR_MODEL)

        self.agent = AutoGPT.from_llm_and_tools(
                    ai_name="BFSAgent",
                    ai_role="Assistant",
                    tools=tools,
                    llm=agent_llm,
                    memory=vectorstore.as_retriever(),
                    max_iterations=2*self.num_iterations
                )
        # Set verbose to be true
        self.agent.chain.verbose = False

    def extract_info_prompt(self, task, data):
        extract_start = time.time()
        encoded = self.enc.encode(data)
        print("Length of content: ", len(encoded))
        print(self.last_search_thought)
        print(self.last_selection_thought)

        if len(encoded) > 100000:
            data = self.enc.decode(encoded[:100000])

        prompt = """WEBSITE CONTENT: {data}\n\n\n
        The above website was chosen with the following motivation by the web navigator when searching the web: {search_motivation}
        
        You must follow the navigator's current motivation to identify the relevant information from the provided content. Note that the relevant information can directly or even partially address what the navigator is looking or even provide more context to later navigator searches. You must return the relevant information in the form of a list of paragraphs. Each paragraph should NOT be longer than 8 sentences.\n You can return upto two paragraphs ONLY. 

        Your goal is to be an EXTRACTOR, i.e. you must only identify and extract relevant facts from the website content. However, you MUST NOT draw any inferences on your own based on the website content or directly try to answer the navigator's motivation. You can choose to merge or combine these facts as needed since you can return only two paragraphs. 
        
        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
        Response Format: 
        {{ 
            "thoughts": "Concise justification for why the information was deemed relevant",
            "paragraphs": [list of relevant paragraphs according to the navigator's motivation]
        }}"""

        inp_prompt = prompt.format(data=data, search_motivation=self.last_search_thought)
        messages = [
                        {"role": "system", "content": """You are an assistant helping an information aggregation process. You are working with a web nagivator that iteratively searches for information. Your goal is identify and extract any relevant information from the website content that the web navigator has provided to you in the current iteration. DO NOT RESPOND with your own knowledge, only respond BASED on information provided in the text."""},
                        {"role": "user", "content": inp_prompt}
                    ]
        llm = ChatOpenAI(model=self.EXTRACTOR_MODEL, max_tokens=2000, temperature=0)
        structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
        with get_openai_callback() as cb:
            response = structured_llm.invoke(messages)  
        self.extractor_cost += float(cb.total_cost)
        self.extractor_time += time.time() - extract_start

        if response["parsing_error"]:
            return [response["raw"].content], response["raw"].content, True
        else:
            data = response["parsed"]
            if type(data).__name__ == "dict":
                if len(data) == 2:
                    value = data["paragraphs"]
                    if type(value).__name__ == "list":
                        return value, response["parsed"], False
                    else:
                        return [value], response["parsed"], False
                else:
                    return [data]
            elif type(data).__name__ == "list":
                return data, response["parsed"], False
            else:
                return [data], response["parsed"], False

    def search(self, query:str) -> List[Dict]:
        assistant_reply = self.agent.chat_history_memory.messages[-1].content
        try:
            parsed = json.loads(assistant_reply, strict=False)
            self.last_search_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(assistant_reply)
            try:
                parsed = json.loads(preprocessed_text, strict=False)   
                self.last_search_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]             
            except Exception:
                print("Error parsing search thought")

        search_start = time.time()
        """Useful for when you need to gather information from the web. You should ask targeted questions"""
        results = self.search_tool.results(query + " site:en.wikipedia.org")
        snippets = list()
        for r in results["organic"][:5]:
            snippets.append({
                "title": r["title"],
                "url": r["link"],
                "snippet": r["snippet"]
            })
        self.search_time += time.time() - search_start
        self.current_query = query
        return snippets

    def aggregator(self, data):
        aggregator_start = time.time()
        if len(self.aggregated_data) > 0:
            aggregated_list = ""
            for pid, text in enumerate(self.aggregated_data):
                aggregated_list += "ID: E" + str(pid+1) + " Text: " + text + "\n\n"
        else:
            aggregated_list = "None"

        provided_list = ""
        for pid, text in enumerate(data):
            provided_list += "ID: P" + str(pid+1) + " Text: " + text + "\n\n"

        previous_iterations = ""
        for c, iter in enumerate(self.previous_iterations):
            previous_iterations += """Iteration {i}: Thoughts - {t}\nFeedback - {f}\n\n""".format(i=str(c+1), t=iter["thoughts"], f=iter["feedback"])

        system_prompt = """You are an information aggregation assistant designed to aggregate relevant information across multiple iterations for a given user query. Your goal at every iteration is to identify any relevant information that can help answer user query. If the information does not DIRECTLY answer the user query, you can still consider the information if it provides context that is useful for later iterations in order to answer the user query. Make sure to not gather information that is duplicate, i.e. do not add redudant information into what you have already aggregated. You should stop aggregating when you have enough information to answer the user query. REMEMBER that your goal is ONLY to aggregate relevant information and NOT to generate the FINAL answer to the user query."""
        system_input = system_prompt

        prompt = """You will be provided with a set of passages collected from a website by a navigator assistant. You need to decide whether any of the provided information should be added into the aggregated information list. You can choose to add the information if it can address the user query. If the information does not DIRECTLY address the user query, you can still aggregate the information if it provides context that is useful for later iterations in order to answer the user query. You also have the option to ignore and not aggregate any of the provided passages into the aggregated information list. 
        
        Further, you should provide feedback to the navigator assistant on what specific information to look for next. If the information needs to be gathered in multiple steps, you can break it down to multiple steps and sequentially instruct the navigator. The navigator assistant cannot see the information aggregated so far, so you make sure the feedback is clear and does not have any ambiguity. Make sure to refer to any entities by their full names when providing the feedback so that the navigator knows what you are referring to. You should instruct the navigator to terminate if enough information to answer the user query has been aggregated.

        You have a maximum of {num_iterations} iterations overall, after which the information aggregated will be automatically returned. You are also provided access to your thoughts and navigator feedback from previous iterations, so that you don't repeat yourself. In addition, the navigator's thoughts are also shown to give additional context for why the provided information was extracted.
        
        Current Iteration Counter: {counter}

        User Task: {user_task}

        Previous Iterations: {previous_iterations}

        Navigator Thoughts: {navigator_motivation}

        Information Aggregated so far: 
        {aggregated_list}

        Provided information: 
        {provided_list}  

        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads
        Response Format: 
        {{
            "thoughts": "Your step-by-step reasoning for what actions to perform based on the provided information",
            "actions": [list of actions (generated as a string) to perform. Allowed actions are: REPLACE(existing_id, provided_id) if passage existing_id in aggregated information should be replaced by passage provided_id from provided information and ADD(provided_id) if passage provided_id should be added to aggregated information],
            "feedback": "Feedback to return to the navigator assistant n what specific information to look for next. The navigator assistant does not have access to the information aggregated, so be clear in your feedback. Also let the navigator assist know how many more iterations are left."
        }}"""
        inp_prompt = prompt.format(num_iterations=str(self.num_iterations), counter=str(self.counter), user_task=self.user_task, previous_iterations=previous_iterations, navigator_motivation=self.last_search_thought, aggregated_list=aggregated_list, provided_list=provided_list)

        messages = [
                        {"role": "system", "content": system_input},
                        {"role": "user", "content": inp_prompt}
                    ]
        # print(inp_prompt)
        if "claude" in self.AGGREGATOR_MODEL:
            llm = ChatAnthropic(model=self.AGGREGATOR_MODEL, max_tokens=1000, temperature=0.7)
            structured_llm = llm.with_structured_output(include_raw=True)
        elif "gemini" in self.AGGREGATOR_MODEL: 
            llm = ChatGoogleGenerativeAI(model=self.AGGREGATOR_MODEL, max_tokens=1000, temperature=0.7)
            structured_llm = llm.with_structured_output(include_raw=True)
        else:
            llm = ChatOpenAI(model=self.AGGREGATOR_MODEL, max_tokens=1000, temperature=0.7)
            structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
        
        with get_openai_callback() as cb:
            response = structured_llm.invoke(messages)
        self.aggregator_cost += float(cb.total_cost)
        self.aggregator_time += time.time() - aggregator_start

        if response["parsing_error"]:
            return "Aggregator failed to parse the website content. Continue with a different website.", response["raw"].content, True
        else:
            output = response["parsed"]
            self.previous_iterations.append({
                "thoughts": output["thoughts"],
                "feedback": output["feedback"]
            })
            print(output)
            for action in output["actions"]:
                try:
                    if "REPLACE" in action:
                        pattern = r'^REPLACE\(E(\d+),\s*P(\d+)\)$'
                        match = re.match(pattern, action)
                        if match:
                            integer1, integer2 = match.groups()
                            self.aggregated_data[int(integer1)-1] = data[int(integer2) - 1]
                        else:
                            print("Invalid Action: ", action)

                    if "ADD" in action:
                        pattern = r"^ADD\(P(-?\d+)\)$"
                        match = re.match(pattern, action)
                        if match:
                            pindex = int(match.group(1)) - 1  # Convert captured string to integer
                            self.aggregated_data.append(data[pindex])
                        else:
                            print("Invalid Action: ", action)
                except Exception as e:
                    print(e)
            
            return output["feedback"] + " Passages aggregated so far: " + str(len(self.aggregated_data)), response["parsed"], False
        
    def extract(self, url: str) -> List[str]:

        assistant_reply = self.agent.chat_history_memory.messages[-1].content
        try:
            parsed = json.loads(assistant_reply, strict=False)
            self.last_selection_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(assistant_reply)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
                self.last_selection_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
            except Exception:
                print("Error parsing selection thought")

        self.counter += 1
        parse_start = time.time()
        """Useful to extract relevant information from the given url for the current user task"""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.content
            text = BeautifulSoup(html, 'html.parser').get_text()
            cleaned_text = re.sub(r'\n+', '\n', text)
            content = markdownify.markdownify(cleaned_text, heading_style="ATX")
        except Exception as e:
            print(str(e))
            content = ""

        print("URL parsing done")
        self.parse_time += time.time() - parse_start

        if content.strip():
            extracted_data, extractor_log, extractor_parse_error = self.extract_info_prompt(self.user_task, content)
            print("Extracted Data: ", extracted_data)
            if len(extracted_data) > 0:
                aggregator_feedback, aggregator_log, aggregator_parse_error = self.aggregator(extracted_data)
            else:
                aggregator_feedback = "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
                aggregator_log = None
                aggregator_parse_error = False
        else:
            aggregator_feedback =  "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
            aggregator_log = None
            aggregator_parse_error = False
            extractor_parse_error = False
            extractor_log = None
        
        self.aggregator_messages.append({
            "url": url,
            "extractor_log": extractor_log,
            "extractor_parse_error": extractor_parse_error,
            "aggregator_log": aggregator_log,
            "aggregator_parse_error": aggregator_parse_error
        })

        return aggregator_feedback

    def run(self):
        output = dict()

        prompt = """You are an assistant aiding an information aggregation process designed to gather information that can answer a multi-hop user question. 
        
        You are provided access to the web (using the "search" command) which returns websites relevant to your search query. Based on the provided websites, you should then choose to vist (using the "extract" command) website that is most relevant. Along with the website URL, you are also provided with a short snippet from the website that can help to decide whether the website is relevant to visit. 
        You should only visit websites that you think will contain information relevant to user query. If the websites do not contain any relevant information, you can choose to perform a different search. DO NOT visit a website that you have already have visited before. Note that information cannot be directly aggregated based on the search command. You MUST ALSO visit the website using the extract command in order to be able to aggregate the relevant information from it. 
        
        You will work in conjunction with an aggregator assistant (which runs as part of the "extract" command) that keeps track of information aggregated and will give feedback to you on what information to look for next. You can decide to stop if aggregator assitant tells you so or if you keep running into a loop. You can simply terminate at the end with a message saying aggregation is done.

        Note that you can only search for one piece of information at a time. For the multi-hop query, it is good to break it down and gather relevant information sequentially over multiple interations. If the extract command suggests you to search for multiple pieces of information, you should search for each piece sequentially over different iterations.

        Below is the user query. 
        Query: {task}"""

        output["navigator_output"] = self.agent.run([prompt.format(task=self.user_task)])
        output["aggregated_output"] = self.aggregated_data
        navigator_cost = float(self.agent.chain_cost)
        output["cost"] = {
            "total_cost": navigator_cost + self.aggregator_cost + self.extractor_cost,
            "navigator_cost": navigator_cost,
            "aggregator_cost": self.aggregator_cost,
            "extractor_cost": self.extractor_cost
        }
        output["langchain_messages"] = list()
        for m in self.agent.chat_history_memory.messages:
            output["langchain_messages"].append(m.content)

        output["aggregator_messages"] = self.aggregator_messages

        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--navigator_model",
        help="Model to use for navigation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--aggregator_model",
        help="Model to use for aggregation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--extractor_model",
        help="Model to use for extraction",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--num_iterations",
        help="Number of iterations to run the aggregator for",
        default=15,
        type=int
    )
    parser.add_argument(
        "--num_to_aggregate",
        help="Number of passages to aggregate",
        default=5,
        type=int
    )
    parser.add_argument(
        "--inp_path",
        help="Input file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--out_path",
        help="Output file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--log_path",
        help="Log file path",
        required=True,
        type=str
    )
    args = parser.parse_args()

    data = pd.read_csv(args.inp_path, sep='\t')
    output = list()
    log = list()
    for index in range(0, len(data)):
        bfsagent = BFSAgent(args)
        bfsagent.setup(data["Prompt"][index])
        try:
            time_start = time.time()
            output_item = bfsagent.run()
            total_time = time.time() - time_start
            output_item["time"] = {
                "total_time": total_time,
                "navigator_time": total_time - (bfsagent.search_time + bfsagent.parse_time + bfsagent.extractor_time + bfsagent.aggregator_time),
                "extractor_time": bfsagent.extractor_time,
                "aggregator_time": bfsagent.aggregator_time,
                "search_time": bfsagent.search_time,
                "parse_time": bfsagent.parse_time
            }
            output_item["id"] = str(data["ID"][index])
            output_item["wiki_errors"] = deepcopy(bfsagent.wiki_errors)
            output.append(deepcopy(output_item))
        except Exception as e:
            print
            log.append({
                "id": str(data["ID"][index]),
                "error": str(e)
            })
            continue
    
    json.dump(output, open(args.out_path, "w"), indent=4)
    json.dump(log, open(args.log_path, "w"), indent=4)