from evaluate_utils.evaluate_dicts import evaluate_dicts
from evaluate_utils.evaluate_strings import evaluate_strings
from evaluation_evaluator import question_scorer
import json
import numpy as np

# Load the gold data (this will remain constant)
with open('/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/assistantbench_dev/assistantbench_gold.json', 'r') as f1:
    gold_data = json.load(f1)  # Contains the 'gold_answer' and 'task_id'

gold_dict = {entry['task_id']: entry['answer'] for entry in gold_data}

def evaluate_predictions(prediction_file_path):
    """
    Evaluate predictions from the given JSON file against the gold data.

    Args:
        prediction_file_path (str): Path to the JSON file containing predictions.

    Returns:
        list: Accuracies for each task.
        float: Mean accuracy.
    """
    with open(prediction_file_path, 'r') as f:
        prediction_data = json.load(f)  # Contains 'answer' and 'id'

    accuracies = []
    for entry in prediction_data:
        task_id = entry.get('id')
        predictions = entry['answer']
        gold = gold_dict.get(task_id, None)
        if gold is not None:
            accuracy, has_ans = question_scorer(predictions, gold)
            entry['gold'] = gold
            entry['accuracy'] = accuracy

            print(task_id, accuracy, predictions, gold)
            accuracies.append(accuracy)

    # Handle cases where some tasks are missing predictions
    accuracies += [0] * (len(gold_data) - len(prediction_data))

    return accuracies, np.mean(np.array(accuracies))

# Example usage
prediction_file_path = '/home/sagnikm3/infogent/interactive-visual-access/src/assistantbench_evaluator/assistantbench_dev/infogent_gpt4o.json'
accuracies, mean_accuracy = evaluate_predictions(prediction_file_path)
print(accuracies)
print(mean_accuracy)
