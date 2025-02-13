from evaluate_utils.evaluate_dicts import evaluate_dicts
from evaluate_utils.evaluate_strings import evaluate_strings
from evaluation_evaluator import question_scorer
import json
import numpy as np


with open('/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/assistantbench_dev/assistantbench_gold.json', 'r') as f1:
    data1 = json.load(f1)  # Contains the 'gold_answer' and 'task_id'

data2 = []
with open('/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/assistantbench_dev/infogent_gpt4o.json', 'r') as f2:
    data2 = json.load(f2)
    # data2 = json.load(f2)

gold_dict = {entry['task_id']: entry['answer'] for entry in data1}
accuracies = []
for entry in data2:
    task_id = entry.get('id')
    predictions = entry['answer']
    gold = gold_dict[task_id]
    accuracy, has_ans = question_scorer(predictions,gold)
    entry['gold'] = gold
    entry['accuracy'] = accuracy

    print(task_id, accuracy, predictions, gold)
    accuracies.append(accuracy)

accuracies = accuracies + [0]* (len(data1) - len(data2))
print(accuracies)
print(np.mean(np.array(accuracies)))


