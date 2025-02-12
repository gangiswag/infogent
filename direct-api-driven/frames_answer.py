from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from tqdm import tqdm
from collections import Counter
import fanoutqa
import tiktoken
import pandas as pd

def get_prompt(question):
    prompt = (
        f"Answer the following question to the best of your knowledge, and output only your answer. Answer should be concise. Never refuse to answer.\n\n[Question]: {question}"
    )
    return prompt

def get_closed_book_output(args):
    dataset = pd.read_csv(args.data_path, sep='\t')
    output = list()
    for index in tqdm(range(0, len(dataset))):
        llm = ChatOpenAI(model=MODEL, max_tokens=2000, temperature=0)
        prompt = get_prompt(dataset["Prompt"][index])
        messages = [
                    {"role": "user", "content": prompt}
                ]
        response = llm.invoke(messages)  

        output.append({
            "id": str(dataset["ID"][index]),
            "answer": response.content
        })
    json.dump(output, open(args.out_path, "w"), indent=4)
    return output

def get_search_prompt(question, passages):
    prompt = """Based the provided context, answer the following question. Output only your answer. Answer should be concise. Never refuse to answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    context = ""
    for index, passage in enumerate(passages):
        context += str(index + 1) + ") " + passage.strip() + "\n"

    return prompt.format(context=context, question=question)

def get_search_output(args):
    data = json.load(open(args.inp_path))
    dataset = pd.read_csv(args.data_path, sep='\t')
    dataset_map = {str(dataset["ID"][index]): dataset["Prompt"][index] for index in range(0, len(dataset))}
    output = list()
    counts = list()
    for item in tqdm(data):
        counts.append(len(item["aggregated_output"]))
        llm = ChatOpenAI(model=MODEL, max_tokens=2000, temperature=0)
        messages = [
                    {"role": "user", "content": get_search_prompt(dataset_map[item["id"]], item["aggregated_output"])}
                ]
        with get_openai_callback() as cb:
            response = llm.invoke(messages)  

        output.append({
            "id": item["id"],
            "answer": response.content
        })
    
    return output

def run_eval(answers, args):

    output = list()
    dataset = pd.read_csv(args.data_path, sep='\t')

    prompt = """"===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth answer for a given question. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.

===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:"

===Input Data===
- Question: {question}
- Predicted Answer: {predicted}
- Ground Truth Answer: {answer}

===Output Format===
You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
Response Format: 
{{ 
    "Explanation": "(How you made the decision?)",
    "Decision": "TRUE" or "FALSE"
}}"""

    answers = {item["id"]: item["answer"] for item in answers}
    for index in tqdm(range(0, len(dataset)):
        id = str(dataset["ID"][index])
        if id not in answers:
            print("Not present: ", id)
            output.append(
                {                    
                    "id": id,
                    "eval": "false",
                    "present": False
                }
            )
        else:
            llm = ChatOpenAI(model=MODEL, max_tokens=2000, temperature=0)
            structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)            
            messages = [
                        {"role": "user", "content": prompt.format(question=dataset["Prompt"][index], predicted=answers[id], answer=dataset["Answer"][index])}
                    ]
            response = structured_llm.invoke(messages)  

            output.append(
                {                    
                    "id": id,
                    "eval": str(response["parsed"]["Decision"]).lower(),
                    "present": True
                }
            )
    json.dump(output, open(args.score_path, "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer_model",
        help="Model to use for answer generation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--data_path",
        help="Dataset file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--closed_book", 
        action='store_true', 
        help="whether to run closed book setting")
    
    parser.add_argument(
        "--inp_path",
        help="Aggregated Data path",
        type=str
    )
    parser.add_argument(
        "--out_path",
        help="Output Answer file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--score_path",
        help="Output Score file path",
        required=True,
        type=str
    )
    args = parser.parse_args()

    if args.closed_book:
        answers = get_closed_book_output(args)
    else:
        assert args.inp_path
        answers = get_search_output(args)
    
    run_eval(answers, args)