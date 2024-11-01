from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import json
from tqdm import tqdm
from collections import Counter
import fanoutqa

MODEL="gpt-4o-mini"
# MODEL="gpt-4-turbo"

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

def get_output(data):
    output = list()
    for item in tqdm(data):
        llm = ChatOpenAI(model=MODEL, max_tokens=2000, temperature=0)
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
    return output

def run_with_search(data, dataset):
    dataset_map = {item["id"]: item["question"] for item in dataset}
    output = list()
    counts = list()
    for item in tqdm(data):
        counts.append(len(item["aggregated_output"]))
        llm = ChatOpenAI(model=MODEL, max_tokens=4000, temperature=0)
        messages = [
                    {"role": "user", "content": get_search_prompt(dataset_map[item["id"]], item["aggregated_output"])}
                ]
        with get_openai_callback() as cb:
            response = llm.invoke(messages)  

        output.append({
            "id": item["id"],
            "answer": response.content
        })
    # print(Counter(counts))
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


# conda activate bfsagent
# cd /shared/nas2/revanth3/bfsagent/fanoutqa

def run_eval(data_path, out_path, score_path):
    data = json.load(open(data_path))
    qs = [DictToObject(q) for q in data]
    answers = json.load(open(out_path))
    scores = fanoutqa.eval.evaluate(qs,answers)
    json.dump(scores, open(score_path, "w"), indent=4, default=serialize_object)           

if __name__ == "__main__":
    # data = json.load(open("/shared/nas2/revanth3/bfsagent/fanoutqa/fanoutqa/data/fanout-final-dev.json"))
    # output = get_output(data)
    # json.dump(output, open("/shared/nas2/revanth3/bfsagent/exp/fanoutqa/closed_book/gpt-4o-mini.json", "w"), indent=4)

    # run_eval("/shared/nas2/revanth3/bfsagent/fanoutqa/fanoutqa/data/fanout-final-dev.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/closed_book/gpt-4o-mini.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/closed_book/gpt-4o-mini_eval.json")

    dataset = json.load(open("/shared/nas2/revanth3/bfsagent/fanoutqa/fanoutqa/data/fanout-final-dev.json"))
    # data = json.load(open("/shared/nas2/revanth3/bfsagent/exp/fanoutqa/agg_agent/fanout_dev.json"))
    # output = run_with_search(data, dataset)
    # json.dump(output, open("/shared/nas2/revanth3/bfsagent/exp/fanoutqa/agg_agent/fanout_dev_gpt-4-turbo_answer.json", "w"), indent=4)

    # run_eval("/shared/nas2/revanth3/bfsagent/fanoutqa/fanoutqa/data/fanout-final-dev.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/agg_agent/fanout_dev_gpt-4-turbo_answer.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/agg_agent/fanout_dev_gpt-4-turbo_eval.json")

    data = json.load(open("/shared/nas2/revanth3/bfsagent/exp/fanoutqa/mindsearch/fanout_dev.json"))
    output = run_with_search(data, dataset)
    json.dump(output, open("/shared/nas2/revanth3/bfsagent/exp/fanoutqa/mindsearch/fanout_dev_gpt-4o-mini_answer.json", "w"), indent=4)

    run_eval("/shared/nas2/revanth3/bfsagent/fanoutqa/fanoutqa/data/fanout-final-dev.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/mindsearch/fanout_dev_gpt-4o-mini_answer.json", "/shared/nas2/revanth3/bfsagent/exp/fanoutqa/mindsearch/fanout_dev_gpt-4o-mini_eval.json")