from evaluate_utils.evaluate_dicts import evaluate_dicts
from evaluate_utils.evaluate_strings import evaluate_strings
from evaluation_evaluator import question_scorer
import json
import numpy as np


with open('/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/fanoutqa/fanoutqa_gold.json', 'r') as f1, open('/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/fanoutqa/fanoutqa_gpt4turbo.json', 'r') as f2:
    data1 = json.load(f1)  # Contains the 'gold_answer' and 'task_id'
    data2 = json.load(f2)  # Contains 'final_answer' and 'id'

gold_dict = {entry['task_id']: entry['gold_answer'] for entry in data1}
accuracies = []
for entry in data2:
    task_id = entry.get('id')
    predictions = entry['final_answer']
    gold = gold_dict[task_id]
    accuracies.append(evaluate_dicts(predictions,gold))

accuracies = accuracies + [0]* (len(data1) - len(data2))
print(accuracies)
print(np.mean(np.array(accuracies)))


