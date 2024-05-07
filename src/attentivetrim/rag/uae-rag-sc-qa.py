
import os
import json
import time

import dsp
import dspy
from src.attentivetrim.rag.char_chunker import get_chunks_char
from src.attentivetrim.tool.dspy_interface import dspyCOT
from src.attentivetrim.tool.extraction_sample import SingleQuestionOverSample


def get_answer(question, top_n_indices, file_name):
    print(f'Question: {question},top n: {len(top_n_indices)}, Top N Indices: {top_n_indices}, File Name: {file_name}')
    # Assuming BASEDIR and chunk_size are defined globally or are part of the configuration
    chunks = get_chunks_char("/Users/chunwei/pvldb_1-16/16/" + file_name, chunk_char_size=500)
    # Sort the top_n_indices by ascending order
    top_n_indices.sort()
    # Extract the relevant chunks based on top_n_indices
    selected_chunks = [chunks[i] for i in top_n_indices if i < len(chunks)]

    # Concatenate these chunks with "..." in between
    sample = "\n ... ".join(selected_chunks)

    print(f"len(sample): {len(sample)}")
    # print(f"Sample: {sample}")
    # Initialize the cot model (assuming dspyCOT is already imported and configured)
    cot = dspyCOT(SingleQuestionOverSample)

    # Measure the runtime of the prediction
    start_time = time.time()
    pred = cot(question, sample)
    end_time = time.time()

    # Calculate duration
    duration = end_time - start_time

    # Return the answer and the duration
    return pred.answer, duration


rag_file="../data/rag-v16-100-UAE-Large-V1-500.json"

grd_files = ["../data/test_v16_inputfile100-result-What is the pap-0.3-location.json",
                "../data/test_v16_inputfile100-result-What is the aut-0.1-location.json",
                "../data/test_v16_inputfile100-result-What is the mai-location.json"]

# Load JSON file
with open(rag_file, 'r') as rag_file:
    data = json.load(rag_file)

n_values = [1, 5, 10, 20, 30]

# Function to calculate coverage
def calculate_coverage(top_indices, given_array, n):
    top_n_indices = top_indices[:n]
    covered = len(set(given_array) & set(top_n_indices))
    total = len(given_array)
    return covered / total

# Process each file and question
results = {}
total = 0

dsp.modules.cache_utils.cache_turn_on = False
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai_key = os.environ['OPENAI_API_KEY']
turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
dspy.settings.configure(lm=turbo)

for file_obj in data['files']:
    total += int(file_obj['total_chars'])
    print(f'File: {file_obj["file"]}')
    for i, question in enumerate(data['questions']):
        print(f'Question: {question}')

        if question not in results:
            results[question] = {n: [] for n in n_values}
        top = file_obj['results'][i]['top_30']
        top_30 = [entry[1] for entry in top]
        for n in n_values:
            top_n_indices = top_30[:n]
            answer, duration = get_answer(question, top_n_indices, file_obj["file"])
            if question not in results:
                results[question] = {}
            if n not in results[question]:
                results[question][n] = []
            results[question][n].append({
                "file": file_obj["file"],
                "result": answer,
                "duration": duration
            })
            print(f'Question: {question}, N: {n}, Answer: {answer}, Duration: {duration:.2f}s')

avg_chars = total / len(data['files'])
print(f'Average number of characters: {avg_chars:.2f}')


# iterate over each questions and n values
for question in results:
    for n in results[question]:
        ratio = f'{n * 500.0 / avg_chars:.2f}'
        json_results = {
            "question": question,
            "files": results[question][n]
        }
        open(f'../data/rag-results-v16-100-UAE-Large-V1-500-{question[:15]}-{ratio}.json', 'w').write(json.dumps(json_results, indent=4))
        print(f'File saved: rag-results-v16-100-UAE-Large-V1-500-{question[:15]}-{ratio}.json')