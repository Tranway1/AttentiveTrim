import json
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance

BASEDIR = "/home/gridsan/cliu/"
DATASET_MAP = {
    "alpaca": "alpaca",
    "google-qa": "google-qa",
    "alpaca-llama": "profiling-Meta-Llama-3-8B-Instruct"
}

dataset = DATASET_MAP["alpaca-llama"]


# Load JSON data
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data['records']


# Embedding function using SentenceTransformer
def compute_embeddings(model, texts, batch_size=10):
    all_embeddings = []
    for start_index in range(0, len(texts), batch_size):
        print("processing batch starting at index: ", start_index)
        batch_texts = texts[start_index:start_index + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)


# Calculate cosine similarity and find top 50 similar prompts
def find_similar_prompts(query_idx, encoded_prompts, records):
    query_vec = encoded_prompts[query_idx]
    similarities = []
    for idx, vec in enumerate(encoded_prompts):
        if idx != query_idx:
            sim = 1 - distance.cosine(query_vec, vec)
            similarities.append((sim, idx))
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_50 = similarities[:50]
    return [(records[idx]['record_id'], records[idx]['prompt'], records[idx]['iteration_count'], sim) for sim, idx in
            top_50]


# Main processing function
def process_prompts(file_path, model):
    records = load_json_data(file_path)
    prompts = [rec['prompt'] for rec in records if rec['prompt']]

    # Check if embeddings file exists, load or compute embeddings
    embeddings_file = BASEDIR + f"probing/{dataset}-length-sfr-embed.npy"
    if os.path.exists(embeddings_file):
        encoded_prompts = np.load(embeddings_file)
    else:
        encoded_prompts = compute_embeddings(model, prompts)
        np.save(embeddings_file, encoded_prompts)

    results = []
    total_entries = min(1000, len(records))
    for i in range(total_entries):
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
    return results


# Initialize the SentenceTransformer model
model_name = BASEDIR + "hf/SFR-Embedding-Mistral"
model = SentenceTransformer(model_name, device='cuda')

# Process prompts and save results
file_path = BASEDIR + f'probing/{dataset}_length.json'
output_results = process_prompts(file_path, model)
output_path = BASEDIR + f"probing/{dataset}-length-sfr-top50.json"
with open(output_path, 'w') as file:
    json.dump({"total_entries": len(output_results), "results": output_results}, file, indent=4)