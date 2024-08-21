import json
import os

from angle_emb import AnglE
from scipy import spatial
import numpy as np

BASE = "/home/gridsan/cliu/"
BASEDIR= f"{BASE}probing/"
# BASEDIR= "/Users/chunwei/research/llm-scheduling/"

DATASET_MAP = {
    "alpaca": "alpaca",
    "google-qa": "google-qa",
    "alpaca-llama": "profiling-Meta-Llama-3-8B-Instruct"
}

dataset = DATASET_MAP["alpaca-llama"]
print(f"Processing dataset: {dataset}")
# Load prompts from JSON file
def load_prompts(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['records']

# Encode prompts using the AnglE model
def encode_prompts(prompts, model):
    encoded_prompts = []
    # if embedding file exists, load it
    if os.path.exists(f'{BASEDIR}{dataset}-length-uae-embed.npy'):
        encoded_array = np.load(f'{BASEDIR}{dataset}-length-uae-embed.npy', allow_pickle=True)
        encoded_prompts = list(encoded_array)
        print("encoded prompts loaded from file")
        return encoded_prompts
    for prompt in prompts:
        if prompt is not None:
            # print(prompt)
            encoded = model.encode(prompt)
            print("encoded: ", encoded)
            encoded_prompts.append(encoded)
        else:
            encoded_prompts.append(None)

    # Convert list of arrays to a single NumPy array for saving
    encoded_array = np.array(encoded_prompts, dtype=object)

    # Save the encoded prompts to a local file
    save_encoded_prompts_to_file(encoded_array, f'{BASEDIR}{dataset}-length-uae-embed.npy')
    print("length of encoded prompts: ", len(encoded_prompts))
    return encoded_prompts

def save_encoded_prompts_to_file(encoded_array, filename):
    np.save(filename, encoded_array)
    print(f"Encoded prompts have been saved to {filename}")

# Calculate cosine similarity and find top 50 similar prompts
def find_similar_prompts(query_idx, encoded_prompts, records):
    similarities = []
    query_vec = encoded_prompts[query_idx]
    # print("query_vec dimension: ", query_vec[0].shape)
    if query_vec is None:
        return []
    for idx, vec in enumerate(encoded_prompts):
        if vec is not None and idx != query_idx:
            # print("vec dimension: ", vec[0].shape)
            similarity = 1 - spatial.distance.cosine(query_vec[0], vec[0])
            similarities.append((similarity, idx))
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_50 = similarities[:50]
    return [(records[idx]['record_id'], records[idx]['prompt'], records[idx]['iteration_count'], sim) for sim, idx in top_50]

# Main function to process the data
def process_prompts(file_path, model):
    records = load_prompts(file_path)
    encoded_prompts = encode_prompts([rec['prompt'] for rec in records], model)
    results = []
    total_entries = 1000

    for i in range(min(total_entries, len(records))):
        if records[i]['prompt']:
            top_50 = find_similar_prompts(i, encoded_prompts, records)
            average_count = sum(x[2] for x in top_50) / len(top_50) if top_50 else 0
            average_5 = sum(x[2] for x in top_50[:5]) / 5 if top_50 else 0
            average_10 = sum(x[2] for x in top_50[:10]) / 10 if top_50 else 0
            average_20 = sum(x[2] for x in top_50[:20]) / 20 if top_50 else 0
            average_30 = sum(x[2] for x in top_50[:30]) / 30 if top_50 else 0
            average_40 = sum(x[2] for x in top_50[:40]) / 40 if top_50 else 0

            result = {
                "record_id": records[i]['record_id'],
                "prompt": records[i]['prompt'],
                "iteration_count": records[i]['iteration_count'],
                "top_50": {
                    "average_5": average_5,
                    "average_10": average_10,
                    "average_20": average_20,
                    "average_30": average_30,
                    "average_40": average_40,
                    "average_length": average_count,
                    "records": [{"record_id": x[0], "prompt": x[1], "iteration_count": x[2], "score": x[3]} for x in top_50]
                }
            }
            results.append(result)

    output = {
        "total_entries": total_entries,
        "results": results
    }
    return output



# Load the model
model_name = BASE+"hf/UAE-Large-V1"
angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').cuda()

# Process the prompts and print the results
file_path = f'{BASEDIR}{dataset}_length.json'
output_data = process_prompts(file_path, angle)
top_50_path =f"{BASEDIR}{dataset}-length-uae-top50.json"
with open(top_50_path, 'w') as file:
    json.dump(output_data, file, indent=4)
