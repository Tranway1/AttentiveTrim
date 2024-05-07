import json
import matplotlib.pyplot as plt
import os


# Define multiple arrays of JSON files, each array is a different series
models = ["SFR", "UAE-Large-V1"]
model = models[0]
json_file_series = {
    "title": [
        f"rag-results-v16-100-{model}-500-What is the pap-0.01-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the pap-0.03-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the pap-0.06-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the pap-0.13-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the pap-0.19-acc-full.json"
    ],
    "authors": [
        f"rag-results-v16-100-{model}-500-Who are the aut-0.01-acc-full.json",
        f"rag-results-v16-100-{model}-500-Who are the aut-0.03-acc-full.json",
        f"rag-results-v16-100-{model}-500-Who are the aut-0.06-acc-full.json",
        f"rag-results-v16-100-{model}-500-Who are the aut-0.13-acc-full.json",
        f"rag-results-v16-100-{model}-500-Who are the aut-0.19-acc-full.json"
    ],
    "contribution": [
        f"rag-results-v16-100-{model}-500-What is the mai-0.01-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the mai-0.03-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the mai-0.06-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the mai-0.13-acc-full.json",
        f"rag-results-v16-100-{model}-500-What is the mai-0.19-acc-full.json"
    ]
}

# Base directory where JSON files are stored
base_dir = "../../data/rag-qa/"


# Function to extract matched ratio from a JSON file
def extract_matched_ratio(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        total_matches = data['total_matches']
        total_files = data['total_files']
        matched_ratio = total_matches / total_files
    return matched_ratio


# Plotting function for each series
def plot_series(json_files, series_label):
    x_values = []
    y_values = []
    for json_file in json_files:
        # Extract the parameter value from the file name
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        x_values.append(param_value)

        # Calculate the matched ratio
        matched_ratio = extract_matched_ratio(base_dir + json_file)
        y_values.append(matched_ratio)

    # Plotting the line chart for the current series
    plt.plot(x_values, y_values, marker='o', linestyle='-', label=series_label)


# Process and plot each series
for series_label, json_files in json_file_series.items():
    plot_series(json_files, series_label)
plt.xlim(0, 0.2)
# Customize the plot
plt.xlabel('Input proportion of the paper')
plt.ylabel('Matched Ratio')
plt.title(f'Accuracy for paper questions with {model} model')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig(f'../../figures/figure/line-questions-wide-{model}.pdf')