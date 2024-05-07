import json
import matplotlib.pyplot as plt
import numpy as np


def extract_covered_chunks(start_char: int, end_char: int, total_chars: int, chunk_size: int = 500) -> list:
    """
    Extracts the indices of chunks that cover a specified range of characters.

    :param start_char: The starting character index (inclusive).
    :param end_char: The ending character index (inclusive).
    :param total_chars: The total number of characters.
    :param chunk_size: The number of characters each chunk contains.
    :return: A list of chunk indices that cover the range from start_char to end_char.
    """
    # Calculate the starting and ending chunk indices
    start_chunk_index = start_char // chunk_size
    end_chunk_index = end_char // chunk_size

    # Generate the list of chunk indices
    covered_chunks = list(range(start_chunk_index, end_chunk_index + 1))

    return covered_chunks



rag_file="../data/rag-v16-100-UAE-Large-V1-500.json"

grd_files = ["../data/test_v16_inputfile100-result-What is the pap-0.3-location.json",
                "../data/test_v16_inputfile100-result-What is the aut-0.1-location.json",
                "../data/test_v16_inputfile100-result-What is the mai-location.json"]

# Load JSON file
with open(rag_file, 'r') as file:
    data = json.load(file)

# Load grd files as a list of json objects
grd_data = []
for grd_file in grd_files:
    with open(grd_file, 'r') as file:
        grd_data.append(json.load(file))

# Given array and n values
given_array = [5, 6]
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
for file in data['files']:
    total += int(file['total_chars'])
    print(f'File: {file["file"]}, Total Chars: {file["total_chars"]}')
    for i, question in enumerate(data['questions']):
        print(f'Question: {question}')
        f_grds = grd_data[i]['files']
        for grd in f_grds:
            if file['file'] == grd['file']:
                start_char = grd['start']
                end_char = grd['end']
                total_chars = file['total_chars']
                given_array = extract_covered_chunks(start_char, end_char, total_chars)
                break

        if question not in results:
            results[question] = {n: [] for n in n_values}
        top = file['results'][i]['top_30']
        top_30 = [entry[1] for entry in top]
        for n in n_values:
                coverage_ratio = calculate_coverage(top_30, given_array, n)
                print(f'Question: {question}, N: {n}, Coverage Ratio: {coverage_ratio:.2f}, given_array: {given_array}')
                results[question][n].append(coverage_ratio)

# Calculate average coverage for each question and n
average_results = {question: {n: np.mean(ratios) for n, ratios in n_results.items()} for question, n_results in results.items()}

# Plotting
fig, ax = plt.subplots()
n_groups = len(data['questions'])
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

avg_chars = total / len(data['files'])

print(f'Average number of characters: {avg_chars:.2f}')
for i, n in enumerate(n_values):
    coverage_ratios = [average_results[question][n] for question in data['questions']]
    plt.bar(index + i * bar_width, coverage_ratios, bar_width, alpha=opacity, label=f'Top {n} ({n*500.0/avg_chars:.2f})')

plt.xlabel('Questions')
plt.ylabel('Average Hit Ratio')
plt.title('UAE Average Recall by Question and Top N')
plt.xticks(index + bar_width, data['questions'], rotation=5)
plt.legend()

plt.tight_layout()
plt.savefig('../data/rag-v16-100-UAE-Large-V1.pdf')