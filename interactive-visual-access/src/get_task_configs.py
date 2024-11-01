import json
import os
import random

dataset_root = "/Users/mikewang/Desktop/SP2024/CS598DHT-ConversationalAI/project/BFSWebAgent/data/sampled_dataset"
# wildbench_data = json.load(open(os.path.join(dataset_root, "sampled_wildbench.json")))
# infobench_data = json.load(open(os.path.join(dataset_root, "sampled_infobench.json")))

wildbench_data = json.load(open(os.path.join(dataset_root, "sampled_64_wildbench.json")))
infobench_data = json.load(open(os.path.join(dataset_root, "sampled_64_infobench.json")))

k = 10

# filter wildbench data
wildbench_data_filtered = []
for item in wildbench_data:
    if item['category'] == 'Information seeking':
        if 'Advice seeking' in item['subset'] \
            or 'Information seeking' in item['subset']:
            # remove too long instructions (containing too much information)
            if len(item['instruction']) > 1000:
                continue
            
            instruction = item['instruction'].strip('.').strip('\n')
            item = {
                "confirmed_task": f"{instruction}. {item['input']}",
                "website": "https://www.google.com/",
                "task_id": "widlbench__" + item['id'],
            }
            wildbench_data_filtered.append(item)

print(f"wildbench_data_filtered: {len(wildbench_data_filtered)}")
print(f"sample first: {k}")
wildbench_data_filtered = wildbench_data_filtered[:k]
output_wildbench_path = os.path.join("../data", f"online_tasks/wildbench_sampled_64_{k}.json")
with open(output_wildbench_path, 'w') as f:
    json.dump(wildbench_data_filtered, f, indent=4)


# filter infobench data
infobench_data_filtered = []
for item in infobench_data:
    if item['subset'] == 'Hard_set':
        instruction = item['instruction'].strip('.').strip('\n')
        item = {
            "confirmed_task": f"{instruction}. {item['input']}",
            "website": "https://www.google.com/",
            "task_id": "infobench__" + item['id'],
        }
        infobench_data_filtered.append(item)

print(f"infobench_data_filtered: {len(infobench_data_filtered)}")
print(f"sample first: {k}")
infobench_data_filtered = infobench_data_filtered[:k]
output_infobench_path = os.path.join("../data", f"online_tasks/infobench_sampled_64_{k}.json")
with open(output_infobench_path, 'w') as f:
    json.dump(infobench_data_filtered, f, indent=4)




