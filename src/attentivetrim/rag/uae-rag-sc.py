import json
import time
from scipy import spatial

from angle_emb import AnglE, Prompts
from char_chunker import get_chunks_char

# Define the questions
QUESTIONS = [
    "What is the paper title?",
    "Who are the authors of the paper?",
    "What is the main contribution of the paper?"
]
BASEDIR = "/home/gridsan/cliu/"

# Load the model
model_name = BASEDIR+"hf/UAE-Large-V1"
angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').cuda()

# Load the list of files
list_file = "../data/test_v16_inputfile100.txt"
with open(list_file) as f:
    list_of_files = [line.strip() for line in f.readlines()]

# Define chunk size
chunk_size = 500

# Prepare the output structure
output = {
    "questions": QUESTIONS,
    "model": model_name,
    "chunk_size": chunk_size,
    "files": []
}

# Process each file
for file_path in list_of_files:
    chunks = get_chunks_char(BASEDIR+"pvldb_1-16/16/"+file_path, chunk_char_size=chunk_size)
    doc_vecs = angle.encode(chunks)
    agg_time = 0
    file_info = {
        "file": file_path.split('/')[-1],
        "total_chars": sum(len(chunk) for chunk in chunks),
        "total_chunks": len(chunks),
        "runtime": agg_time,
        "results": []
    }

    for question in QUESTIONS:
        start_time = time.time()
        qv = angle.encode(Prompts.C.format(text=question))
        res = []
        scores = []
        for idx, dv in enumerate(doc_vecs):
            similarity = 1 - spatial.distance.cosine(qv[0], dv)
            res.append((similarity, idx, chunks[idx]))
            scores.append(similarity)

        # Sort by similarity
        res.sort(key=lambda x: x[0], reverse=True)

        runtime = time.time() - start_time
        agg_time += runtime
        question_result = {
            "question": question,
            "top_30": [(sim, idx, text) for sim, idx, text in res[:30]],
            "runtime": runtime,
            "scores": scores,
        }
        file_info["results"].append(question_result)
        file_info["runtime"] = agg_time

    output["files"].append(file_info)

# Write to JSON file
output_file = f'../data/rag-v16-100-{model_name.split("/")[-1]}-{chunk_size}.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)