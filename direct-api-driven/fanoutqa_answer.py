from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import json
from tqdm import tqdm
import fanoutqa
import argparse

def get_prompt(question):
    prompt = (
        "Answer the following question to the best of your knowledge, and output only your answer. Never refuse to answer. If the answer is a list, output one on"
        f" each line. Current date: 11-20-2023.\n\n[Question]: {question}"
    )
    return prompt

def get_search_prompt(question, passages):
    prompt = """Based the provided context, answer the following question. Output only your answer. Never refuse to answer. If the answer is a list, output one on each line. For all time sensitive questions, consider current date: 11-20-2023.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    context = ""
    for index, passage in enumerate(passages):
        context += str(index + 1) + ") " + passage.strip() + "\n"
    
    return prompt.format(context=context, question=question)

def get_closed_book_output(args):
    dataset = json.load(open(args.data_path))
    output = list()
    for item in tqdm(dataset):
        llm = ChatOpenAI(model=args.answer_model, max_tokens=4000, temperature=0)
        prompt = get_prompt(item["question"])
        messages = [
                    {"role": "user", "content": prompt}
                ]
        with get_openai_callback() as cb:
            response = llm.invoke(messages)  

        output.append({
            "id": item["id"],
            "answer": response.content
        })
    json.dump(output, open(args.out_path, "w"), indent=4)
    return output

def get_search_output(args):
    data = json.load(open(args.inp_path))
    dataset = json.load(open(args.data_path))
    dataset_map = {item["id"]: item["question"] for item in dataset}
    output = list()
    counts = list()
    for item in tqdm(data):
        counts.append(len(item["aggregated_output"]))
        llm = ChatOpenAI(model=args.answer_model, max_tokens=4000, temperature=0)
        messages = [
                    {"role": "user", "content": get_search_prompt(dataset_map[item["id"]], item["aggregated_output"])}
                ]
        with get_openai_callback() as cb:
            response = llm.invoke(messages)  

        output.append({
            "id": item["id"],
            "answer": response.content
        })

    json.dump(output, open(args.out_path, "w"), indent=4)
    return output

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def serialize_object(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_eval(answers, args):
    data = json.load(open(args, args.data_path))
    qs = [DictToObject(q) for q in data]
    scores = fanoutqa.eval.evaluate(qs, answers)
    json.dump(scores, open(args.score_path, "w"), indent=4, default=serialize_object)           

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




