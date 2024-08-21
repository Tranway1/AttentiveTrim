import json

from datasets import DatasetDict

# model_name = "google-qa"
model_name = "alpaca"
# Assuming data_dir is defined as the directory where the dataset is stored
data_dir = f"/Users/chunwei/Downloads/{model_name}_processed2"

# Load the dataset from the specified directory
test_data = DatasetDict.load_from_disk(data_dir)['test']
train_data = DatasetDict.load_from_disk(data_dir)['train']

output= {}
output['records'] = []
cnt = 0
if model_name == "google-qa":
    for rec in test_data:
        output['records'].append({
            "record_id": cnt,
            "original_id": rec['id'],
            "prompt": rec['question'],
            "iteration_count": rec['output_token_len']
        })
        cnt += 1

    output['total_test'] = cnt

    for rec in train_data:
        output['records'].append({
            "record_id": cnt,
            "original_id": rec['id'],
            "prompt": rec['question'],
            "iteration_count": rec['output_token_len']
        })
        cnt += 1
    output['total_train'] = cnt - output['total_test']
    output['total'] = cnt



else:
    for rec in test_data:
        output['records'].append({
            "record_id": cnt,
            "prompt": rec['instruction_input_text'],
            "iteration_count": rec['output_token_len']
        })
        cnt += 1

    output['total_test'] = cnt
    for rec in train_data:
        output['records'].append({
            "record_id": cnt,
            "prompt": rec['instruction_input_text'],
            "iteration_count": rec['output_token_len']
        })
        cnt += 1
    output['total_train'] = cnt - output['total_test']
    output['total'] = cnt

out_dir = f"/Users/chunwei/research/llm-scheduling/{model_name}_length.json"
with open(out_dir, 'w') as f:
    json.dump(output, f, indent=4)


