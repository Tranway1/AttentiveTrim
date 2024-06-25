import json
import os
import time
import dsp
import dspy
import numpy as np
from src.attentivetrim.rag.char_chunker import get_chunks_char
from src.attentivetrim.tool.dspy_interface import dspyCOT
from src.attentivetrim.tool.extraction_sample import SingleQuestionOverSample

def get_answer(question, top_n_indices, file_name):
    chunks = get_chunks_char("/Users/chunwei/pvldb_1-16/16/" + file_name, chunk_char_size=500)
    top_n_indices.sort()
    selected_chunks = [chunks[i] for i in top_n_indices if i < len(chunks)]
    sample = "\n ... ".join(selected_chunks)
    print(f"len(sample): {len(sample)}")
    cot = dspyCOT(SingleQuestionOverSample)
    start_time = time.time()
    pred = cot(question, sample)
    end_time = time.time()
    duration = end_time - start_time
    return pred.answer, duration

rag_file = "../data/rag-tr/rag-tr-v16-100-SFR-Embedding-Mistral-500-0.3.json"
grd_files = [
    "../data/test_v16_inputfile100-result-What is the pap-0.3-location.json",
    "../data/test_v16_inputfile100-result-What is the aut-0.1-location.json",
    "../data/test_v16_inputfile100-result-What is the mai-location.json"
]

with open(rag_file, 'r') as file:
    data = json.load(file)

text_ratio = float(data['ratio'])
n_values = [1, 5, 10, 20, 30]
results = {}
total = 0
dsp.modules.cache_utils.cache_turn_on = False
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai_key = os.environ['OPENAI_API_KEY']
turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
dspy.settings.configure(lm=turbo)


for file_obj in data['files']:
    total += (int(file_obj['total_chars'])/text_ratio)
    for i, question in enumerate(data['questions']):

        if question not in results:
            results[question] = {n: [] for n in n_values}
        top_30 = file_obj['top_30'][i]  # Adjusted for new JSON structure
        for n in n_values:
            cur_n = min(n, len(top_30))
            top_n_indices = top_30[:cur_n]
            answer, duration = get_answer(question, top_n_indices, file_obj["file"])
            if n not in results[question]:
                results[question][n] = []
            results[question][n].append({
                "file": file_obj["file"],
                "result": answer,
                "duration": duration
            })
            print(f'Question: {question}, N: {cur_n}, Answer: {answer}, Duration: {duration:.2f}s')

avg_chars = total / len(data['files'])
print(f'Average number of characters: {avg_chars:.2f}')

# Save results to JSON files
for question, question_results in results.items():
    for n, files_results in question_results.items():
        ratio = f'{n * 500.0 / avg_chars:.2f}'
        output_filename = f'../data/rag-tr/rag-tr-results-v16-100-SFR-500-{question[:15]}-{ratio}.json'
        with open(output_filename, 'w') as outfile:
            json.dump({
                "question": question,
                "files": files_results
            }, outfile, indent=4)
        print(f'Results saved to {output_filename}')