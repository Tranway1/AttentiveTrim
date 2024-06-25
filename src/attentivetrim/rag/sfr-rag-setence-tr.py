import torch
from sentence_transformers import SentenceTransformer, util
import time
import json

from char_chunker import get_chunks_char

# Define the questions and task
QUESTIONS = [
    "What is the paper title?",
    "Who are the authors of the paper?",
    "What is the main contribution of the paper?"
]

HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv"]



# BASEDIR = "/Users/chunwei/"
BASEDIR = "/home/gridsan/cliu/"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

task = 'Given question(s) about a paper, retrieve relevant passages that answer the query'
queries = [get_detailed_instruct(task, q) for q in QUESTIONS]

# Load the list of files
list_file = "../data/test_v16_inputfile100.txt"
with open(list_file) as f:
    list_of_files = [line.strip() for line in f.readlines()]

# Initialize the model
# model_name = "Salesforce/SFR-Embedding-Mistral"
model_name = BASEDIR+"hf/SFR-Embedding-Mistral"
model = SentenceTransformer(model_name, device='cuda')
chunk_size = 500  # Assuming chunks are of 500 characters
ratio = 0.3
k=30

# Prepare the output structure
output = {
    "questions": QUESTIONS,
    "model": model_name,
    "chunk_size": chunk_size,
    "ratio": ratio,
    "files": []
}

# list_of_files=['p1-helt_pm.json']

# Process each file
for file_path in list_of_files:
    chunks = get_chunks_char(BASEDIR+"pvldb_1-16/16/"+file_path, chunk_char_size=chunk_size, start_ratio=0.0, end_ratio=ratio)
    passages = chunks

    # Determine number of batches
    n_batches = 4  # You can set this to any number you want
    batch_size = len(passages) // n_batches
    if len(passages) % n_batches != 0:
        n_batches += 1  # Adjust if the last batch will have fewer elements

    all_scores = []

    try:
        start_time = time.time()

        # Process each batch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(passages))
            passages_batch = passages[start_idx:end_idx]
            input_texts_batch = queries + passages_batch

            # Compute embeddings and scores for the batch
            embeddings_batch = model.encode(input_texts_batch)
            scores_batch = util.cos_sim(embeddings_batch[:len(queries)], embeddings_batch[len(queries):]) * 100
            all_scores.append(scores_batch)

        # Merge scores from all batches
        scores = torch.cat(all_scores, dim=1)

        runtime = time.time() - start_time

        cur_k = min(k, len(chunks))
        # Get top 30 scores
        top_k = scores.topk(cur_k, dim=1)


        file_info = {
            "file": file_path.split('/')[-1],
            "total_chars": sum(len(chunk) for chunk in chunks),
            "total_chunks": len(chunks),
            "runtime": runtime,
            "top_30": [],
            "scores": scores.tolist()
        }

        for i, (_, top_k_idx) in enumerate(zip(queries, top_k.indices)):
            top_chunks = [int(idx) for idx in top_k_idx]
            file_info["top_30"].append(top_chunks)

        output["files"].append(file_info)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        continue

name = model_name.split("/")[-1]
# Write to json file
output_file = f'../data/rag-tr/rag-tr-v16-100-{name}-{chunk_size}-{ratio}.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)