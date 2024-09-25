
import json
import matplotlib.pyplot as plt
import os


QUESTIONS = ["What is the paper title?",
             "What is the authors of the paper?",
             "What is the main contribution of the paper?",
             "What are the baselines used in the evaluation?"]

HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv",
            "../data/frequency-test-baselines.csv"]

GRDS = ["../data/test_v16_inputfile100-result-What is the pap-0.3.json",
            "../data/test_v16_inputfile100-result-What is the aut-0.1.json",
            "../data/test_v16_inputfile100-result-What is the mai.json",
            "../data/test_v16_inputfile100-result-What are the ba.json"]

BUDGETS = [
    [0.001, 0.005, 0.05, 0.3],
    [0.005, 0.01, 0.05, 0.1],
    [0.05, 0.1, 0.15, 0.2, 0.4, 0.9],
    [0.05, 0.1, 0.15, 0.2, 0.4, 0.9]
]

json_file_series = {
    "title": [],
    "authors": [],
    "contribution": [],
    "baselines": []
}

idx = 0
question = QUESTIONS[idx]
bgts = BUDGETS[idx]
mode = "sentence"
enable_fallback = False
for budget in bgts:
    json_file_series["title"].append(f'{mode}/results-{mode}-{enable_fallback}-{question[:15]}-{budget}-acc-full.json')

idx = 1
question = QUESTIONS[idx]
bgts = BUDGETS[idx]
for budget in bgts:
    json_file_series["authors"].append(f'{mode}/results-{mode}-{enable_fallback}-{question[:15]}-{budget}-acc-full.json')

idx = 2
question = QUESTIONS[idx]
bgts = BUDGETS[idx]
for budget in bgts:
    json_file_series["contribution"].append(f'{mode}/results-{mode}-{enable_fallback}-{question[:15]}-{budget}-acc-full.json')

idx = 3
question = QUESTIONS[idx]
bgts = BUDGETS[idx]
for budget in bgts:
    json_file_series["baselines"].append(f'{mode}/results-{mode}-{enable_fallback}-{question[:15]}-{budget}-acc-full.json')



# Base directory where JSON files are stored
base_dir = "/Users/chunwei/PycharmProjects/attentivetrim/src/attentivetrim/data/"

# Fallback JSON files
fallback_files = {
    "title": "section/results-section-True-What is the pap-0.001-acc-full.json",
    "authors": "section/results-section-True-What is the aut-0.005-acc-full.json",
    "contribution": "section/results-section-True-What is the mai-0.05-acc-full.json",
    "baselines": "section/results-section-True-What are the ba-0.05-acc-full.json"
}

# Colors for each series
series_colors = {
    "title": "blue",
    "authors": "green",
    "contribution": "red",
    "baselines": "orange"
}

# Function to extract matched ratio from a JSON file
def extract_matched_ratio(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        total_matches = data['total_matches']
        total_files = data['total_files']
        matched_ratio = total_matches / total_files
    return matched_ratio

def extract_avg_budget(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        avg_budget = data['total_avg_budget']
    return avg_budget

# Plotting function for each series
def plot_series(json_files, series_label):
    x_values = []
    y_values = []
    for json_file in json_files:
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        x_values.append(param_value)
        matched_ratio = extract_matched_ratio(base_dir + json_file)
        y_values.append(matched_ratio)
    plt.plot(x_values, y_values, marker='o', linestyle='-', label=f"{series_label} sentence", color=series_colors[series_label])

# Function to plot fallback data points
def plot_fallback_points():
    fallback_dir = base_dir
    for label, json_file in fallback_files.items():
        matched_ratio = extract_matched_ratio(fallback_dir + json_file)
        param_value = float(extract_avg_budget(fallback_dir + json_file))
        plt.scatter([param_value], [matched_ratio], marker='*', s=100, label=f"{label} section fallback", color=series_colors[label])
        # plt.annotate(f"{label} fallback", (param_value, matched_ratio), textcoords="offset points", xytext=(0,10), ha='center')



# Process and plot each series
for series_label, json_files in json_file_series.items():
    plot_series(json_files, series_label)

# Plot fallback points
plot_fallback_points()

# set x range
# plt.xlim(0.0, 0.21)
# plt.ylim(0.0, 1.0)
# Customize the plot
plt.xlabel('Input proportion of the paper')
plt.ylabel('Matched Ratio')
plt.title('Accuracy for paper questions')
# plot legend with location coordinates
plt.legend()
plt.grid(True)

plt.savefig('../../figures/figure/line-sentence-section-points.pdf')