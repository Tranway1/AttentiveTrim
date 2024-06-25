
import json
import matplotlib.pyplot as plt
import os

# Define multiple arrays of JSON files, each array is a different series
json_file_series = {
    "title": [
        "results-What is the pap-0.001-acc-0.3.json",
        "results-What is the pap-0.005-acc-0.3.json",
        "results-What is the pap-0.05-acc-0.3.json",
        "results-What is the pap-0.3-acc-0.3.json"
    ],
    "authors": [
        "results-What is the aut-0.005-acc-0.1.json",
        "results-What is the aut-0.01-acc-0.1.json",
        "results-What is the aut-0.05-acc-0.1.json",
        "results-What is the aut-0.1-acc-0.1.json"
    ],
    "contribution": [
        "results-What is the mai-0.05-acc.json",
        "results-What is the mai-0.1-acc.json",
        "results-What is the mai-0.15-acc.json",
        "results-What is the mai-0.2-acc-full.json",
        "results-What is the mai-0.4-acc-full.json",
        "results-What is the mai-0.9-acc-full.json"
    ]
}

# Base directory where JSON files are stored
base_dir = "../../data/local-full/"

# Fallback JSON files
fallback_files = {
    "title": "results-fallback-What is the pap-0.001-acc-full.json",
    "authors": "results-fallback-What is the aut-0.005-acc-full.json",
    "contribution": "results-fallback-What is the mai-0.05-acc-full.json"
}

# Colors for each series
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
def plot_series(json_files, series_label):
    x_values = []
    y_values = []
    for json_file in json_files:
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        x_values.append(param_value)
        matched_ratio = extract_matched_ratio(base_dir + json_file)
        y_values.append(matched_ratio)
    plt.plot(x_values, y_values, marker='o', linestyle='-', label=series_label, color=series_colors[series_label])

# Function to plot fallback data points
def plot_fallback_points():
    fallback_dir = "../../data/fallback/"
    for label, json_file in fallback_files.items():
        matched_ratio = extract_matched_ratio(fallback_dir + json_file)
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        plt.scatter([param_value], [matched_ratio], marker='*', s=100, label=f"{label} fallback", color=series_colors[label])
        plt.annotate(f"{label} fallback", (param_value, matched_ratio), textcoords="offset points", xytext=(0,10), ha='center')

# Process and plot each series
for series_label, json_files in json_file_series.items():
    plot_series(json_files, series_label)

# Plot fallback points
plot_fallback_points()

# Customize the plot
plt.xlabel('Input proportion of the paper')
plt.ylabel('Matched Ratio')
plt.title('Accuracy for paper questions')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('../../figures/figure/line-questions-lines-points.pdf')