import os
import json
import argparse

# Load the gold data (this will remain constant)
parser = argparse.ArgumentParser(description='Load a JSON file containing gold answers and task IDs.')
parser.add_argument('--parent_dir', type=str, required=True, help='Path to the JSON file')

args = parser.parse_args()

parent_dir = args.parent_dir

aggregated_data = []

for subfolder in os.listdir(parent_dir):
    subfolder_path = os.path.join(parent_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        json_file_path = os.path.join(subfolder_path, 'info_seeking_results.json')
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            data["id"] = subfolder
            
            aggregated_data.append(data)
        else:
            print(subfolder)

with open('aggregated_data.json', 'w') as outfile:
    json.dump(aggregated_data, outfile, indent=4)


