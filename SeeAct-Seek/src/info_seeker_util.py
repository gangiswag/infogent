
from demo_utils.inference_engine import OpenaiEngine_Text
import json
import time
import langchain
langchain.verbose = False

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

info_seek_prompt_template_single_agent = '''Task: {task}

---
Previously collected information: {previous_info}

---
Previous response: {previous_response}

---
Information from current page: {current_page_info}

---

Check if the current page contains useful information for generating a better response for the task. 
If yes, first aggregate this information with previously collected ones. Then generate a new response based on all the information you collected.
If no, simply copy the previous information and keep the previous response.

Note that you should not include any information that is not directly in the collected information for generating the responses.

Finally, determine if the current response is sufficient for the task. If it is, terminate the process.

You MUST provide your answer in the following JSON format:
{{"aggregated_info": "newly aggregated information", "updated_response": "newly generated response to the task", "terminate": "yes or no", "explanation": "rationale for the decision"}}
'''

info_seek_prompt_template_multi_agent_generator = '''Task: {task}

---
Previously collected information: {previous_info}
---
Information from current page: {current_page_info}

Check if the current page contains useful information for generating a better response for the task. 
If yes, first aggregate this information with previously collected ones. Then generate a new response based on all the information you collected.
If no, simply copy the previous information and keep the previous response.

Note that you should not include any information that is not directly in the collected information for generating the responses.

You have a maximum of {num_iterations} iterations overall, after which the information aggregated will be automatically returned. 
Current Iteration Counter: {counter}

You MUST provide your answer in the following JSON format:
{{"aggregated_info": "newly aggregated information", "updated_response": "newly generated response to the task"}}
'''

info_seek_prompt_template_multi_agent_verifier = '''Task: {task}

---
Current response: {current_response}

Check if the current response is sufficient for the task. If there is no current response or the response is not sufficient, DO NOT terminate the process. Terminate the process only if the response satisfies the task requirements.

You MUST provide your answer in the following JSON format:
{{"terminate": "yes or no", "explanation": "rationale for the decision"}}
'''


import re
def extract_json_object(pred):
    try:
        if "```json" in pred:
            json_string = re.search(r"```json(.*?)```", pred, re.DOTALL)
            if json_string is None:
                # print("ERROR: cannot parse json string")
                pred = None
            else:
                json_string = json_string.group(1).strip().replace("\n", "")
                pred = json.loads(json_string)
        else:
            json_string = re.search(r"\{.*?\}", pred, re.DOTALL)        
            if json_string is None:
                # print("ERROR: cannot parse json string")
                pred = None
            else:
                json_string = json_string.group(0).replace("\n", "")
                pred = json.loads(json_string)
    except:
        # print("ERROR: cannot load json string")
        pred = None
    return pred

class InfoSeeker():
    def __init__(self, type, openai_config):
        self.type = type
        if self.type == "single_agent":
            self.generator = OpenaiEngine_Text(**openai_config,)
            self.verifier = None
        elif self.type == "multi_agent":
            self.generator = OpenaiEngine_Text(**openai_config,)
            self.verifier = OpenaiEngine_Text(**openai_config,)
        
        self.raw_page_info = [{"step":0, "content": "None"}]
        self.aggregated_info = [{"step":0, "content": "None"}]
        self.terminate_decisions = [{"step":0, "content": {"terminate": "no", "explanation": "None"}}]
        self.num_iterations = 5 ##TODO: Sagnik - take input from args
        self.counter = 0
        self.num_to_aggregate = 5
        self.aggregator_cost = 0
        self.AGGREGATOR_MODEL = 'gpt-4o' ## TODO Sagnik - fix hardcoded

    def parse_info_seek_response_single_agent(self, response):
        try:
            response_json_obj = extract_json_object(response)
            return response_json_obj["aggregated_info"], response_json_obj["updated_response"], response_json_obj["terminate"], response_json_obj["explanation"]
        except:
            return None, None, None, None

    def parse_info_seek_response_multi_agent_generator(self, response):
        try:
            response_json_obj = extract_json_object(response)
            return response_json_obj
        except:
            return None, None
    
    def parse_info_seek_response_multi_agent_verifier(self, response):
        try:
            response_json_obj = extract_json_object(response)
            return response_json_obj
        except:
            return None, None


    def aggregator(self, data, search_motivation):
        aggregator_start = time.time()
        aggregated_list = ""
        for pid, text in enumerate(self.aggregated_info):
            aggregated_list += "ID: E" + str(pid+1) + " Text: " + text['content'] + "\n\n"

        provided_list = ""
        for pid, text in enumerate(data):
            provided_list += "ID: P" + str(pid+1) + " Text: " + text + "\n\n"

        system_prompt = """You are an information aggregation assistant designed to aggregate information relevant to the given user task. Your goal is to ensure diversity in the gathered information while ensuring they are ALL relevant to the user task. Make sure to not gather information that is duplicate, i.e. do not add redudant information into what you have already aggregated. You can decide to stop aggregating when you decide you have information to address the user task. Also, you can aggregate only {num_to_aggregate} items in the list and should signal to stop when you have aggregated {num_to_aggregate} items."""
        system_input = system_prompt.format(num_to_aggregate=str(self.num_to_aggregate))

        prompt = """You will be provided with a set of passages collected from a website by a navigator assistant. You need to decide whether any of the provided information should be added into the aggregated information list. You have the option to ignore and not add any of the provided passages into the aggregated information list.  If the information doesnt directly answer the query, you can still aggregate parts of it that might lead to the final answer later.
        The query could be multi-hop, and the navigator might be looking for the information step by step. Hence while deciding, you should consider the navigator's motivation for selecting this particular website.
        Also, you should provide detailed feedback to the navigator assistant on how to proceed next. The navigator assistant cannot see the information aggregated, so be clear and specific in your feedback. 
        If you want the navigator to look for informations step by step, provide a feedback detailing the information to be looked up. Dont give any navigation instruction, but rather what information is to be seeked.
        
        You should instruct the navigator to terminate if enough information has been aggregated.

        You have a maximum of {num_iterations} iterations overall, after which the information aggregated will be automatically returned. 
        Current Iteration Counter: {counter}

        User Task: {user_task}
        Navigator's motivation: {search_motivation}

        Information Aggregated so far: 
        {aggregated_list}

        Provided information: 
        {provided_list}    

        To be successful, it is important to follow the following rules: 
        1. Even if information is not directly relevant to the final answer but answers parts of it, you should aggregate that
        2. You have access to the following actions: REPLACE(existing_id, provided_id) if passage existing_id in aggregated information should be replaced by passage provided_id from provided information and ADD(provided_id) if passage provided_id should be added to aggregated information]

        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads
        Response Format: 
        {{
            "thoughts": "Your step-by-step reasoning for what actions to perform based on the provided information",
            "actions": [list of actions (generated as a string) to perform. Allowed actions are: REPLACE(existing_id, provided_id) if passage existing_id in aggregated information should be replaced by passage provided_id from provided information and ADD(provided_id) if passage provided_id should be added to aggregated information.
            Response Format:
            {{
                "action": one of the allowed actions,
                "reason": a sentence explaining why the action was chosen,
            }}],
            "feedback": "[List of information to be looked up one by one. Each information should be looking for small bits of information. decompose the queries as much as possible.
                        ideally queries should be atomic enough so that it could be searched on wikipedia or google.
                        Ideally, each element in the list should seek one bit of information about a single entity.
                        Instructions should not only contain what information is to be looked for, not where. 
                        The navigator doesnt have access to the aggregated information. So give exact details for what is to be searched.]",

            "terminate": "yes or no. Decide whether enough information has been aggregated. Return yes only if the aggregated paragraphs has all the information to answer the question."
        }}"""
        inp_prompt = prompt.format(num_iterations=str(self.num_iterations), counter=str(self.counter), user_task=self.user_task, aggregated_list=aggregated_list, provided_list=provided_list, search_motivation=search_motivation)
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
        print("here")
        # self.aggregator_time += time.time() - aggregator_start
        # print(response["raw"].content)

        if response["parsing_error"]:
            return "Aggregator failed to parse the website content. Continue with a different website.", response["raw"].content, True, False
        else:
            output = response["parsed"]
            print(output)
            for action in output["actions"]:
                try:
                    if "REPLACE" in action['action']:
                        pattern = r'^REPLACE\(E(\d+),\s*P(\d+)\)$'
                        match = re.match(pattern, action['action'])
                        if match:
                            integer1, integer2 = match.groups()
                            self.aggregated_info[int(integer1)-1]["content"] = data[int(integer2) - 1] + "    " + action["reason"]
                        else:
                            print("Invalid Action: ", action)

                    if "ADD" in action['action']:
                        pattern = r"^ADD\(P(-?\d+)\)$"
                        match = re.match(pattern, action['action'])
                        if match:
                            pindex = int(match.group(1)) - 1  # Convert captured string to integer
                            self.aggregated_info.append({"step": len(self.aggregated_info), "content": data[pindex]+ "    " + action["reason"]})
                        else:
                            print("Invalid Action: ", action)
                except Exception as e:
                    print(e)
            
            self.counter += 1
            if not output["feedback"]:
                output["feedback"] = [" "]
            return output["feedback"][0] + " Passages aggregated so far: " + str(len(self.aggregated_info)), output, output["terminate"], output["feedback"]
    
    def run(self, task_query, new_info, search_motivation):
        self.user_task = task_query
        if self.type == "single_agent":
            # single agent
            #TODO - edit code for single agent as well. 
            prompt = info_seek_prompt_template_single_agent.format(task=task_query, 
                                                                            previous_info=self.aggregated_info[-1]["content"],
                                                                            previous_response=self.responses[-1]["content"],
                                                                            current_page_info=new_info)
            raw_generation = self.generator.generate(prompt=prompt)
            new_aggregated_info, terminate_yon, explanation = self.parse_info_seek_response_single_agent(raw_generation)
        else:
            # multi-agent
            # print("running generator...")
            new_aggregated_info, _, terminate_yon, explanation = self.aggregator(new_info, search_motivation)
            
        
        # update memory                            
        if new_aggregated_info is None:
            self.terminate_decisions.append({"step": len(self.terminate_decisions), "content": self.terminate_decisions[-1]["content"]})
        else:
            self.terminate_decisions.append({"step": len(self.terminate_decisions), "content": {"terminate": terminate_yon, "explanation": explanation[0]}})