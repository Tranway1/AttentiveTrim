
import json
import matplotlib.pyplot as plt
import os

# Define multiple arrays of JSON files, each array is a different series
models = ["SFR", "UAE-Large-V1"]
series_colors = {
    "title": "blue",
    "authors": "green",
    "contribution": "red"
}
# Function to extract matched ratio from a JSON file
def extract_matched_ratio(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        total_matches = data['total_matches']
        total_files = data['total_files']
        matched_ratio = total_matches / total_files
    return matched_ratio

# Plotting function for each series
def plot_series(json_files, series_label, base_dir, marker='o', linestyle='-'):
    x_values = []
    y_values = []
    for json_file in json_files:
        # Extract the parameter value from the file name
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        x_values.append(param_value)
        # Calculate the matched ratio
        matched_ratio = extract_matched_ratio(base_dir + json_file)
        y_values.append(matched_ratio)
    color = None
    for colorkey in series_colors.keys():
        if colorkey in series_label:
            color = series_colors[colorkey]
            break
    # Plotting the line chart for the current series
    if color is None:
        plt.plot(x_values, y_values, marker=marker, linestyle=linestyle, label=series_label)
    else:
        plt.plot(x_values, y_values, marker=marker, linestyle=linestyle, label=series_label, color=color)

# Generate plots for each model
for model in models:
    # Define JSON file series for baseline and RAG-TR
    json_file_series_baseline = {
        "title": [
            f"rag-results-v16-100-{model}-500-What is the pap-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ],
        "authors": [
            f"rag-results-v16-100-{model}-500-Who are the aut-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ],
        "contribution": [
            f"rag-results-v16-100-{model}-500-What is the mai-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ]
    }
    json_file_series_ragtr = {
        "title": [
            f"rag-tr-results-v16-100-{model}-500-What is the pap-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ],
        "authors": [
            f"rag-tr-results-v16-100-{model}-500-Who are the aut-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ],
        "contribution": [
            f"rag-tr-results-v16-100-{model}-500-What is the mai-{p}-acc-full.json" for p in ["0.01", "0.03", "0.06", "0.13", "0.19"]
        ]
    }

    # Process and plot each series for baseline and RAG-TR
    for series_label, json_files in json_file_series_baseline.items():
        plot_series(json_files, f"{series_label} (RAG)", "../../data/rag-qa/")
    for series_label, json_files in json_file_series_ragtr.items():
        plot_series(json_files, f"{series_label} (RAG-TR)", "../../data/rag-tr/", marker='x', linestyle='--')

    plt.xlim(0, 0.2)
    plt.ylim(0.0, 1.0)  # Set y-axis range from 0.0 to 1.0

    # Customize the plot
    plt.xlabel('Input proportion of the paper')
    plt.ylabel('Matched Ratio')
    plt.title(f'Accuracy for paper questions with {model} model')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig(f'../../figures/figure/line-questions-rags-{model}.pdf')
    plt.clf()  # Clear the current figure for the next plot