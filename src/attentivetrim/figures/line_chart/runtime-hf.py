import json
import numpy as np
import matplotlib.pyplot as plt
import os


# Function to parse a JSON file and extract the "duration" values
def extract_duration_values(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        duration_values = [item['duration'] for item in data['files'] if item['result'] != '']
    return duration_values

base_dir = "../../data/hf/"

# Define the mapping of questions to file arrays
json_file_series = {
    "title": [
        "hf-results-What is the pap-0.005.json",
        "hf-results-What is the pap-0.01.json",
        "hf-results-What is the pap-0.05.json",
        "hf-results-What is the pap-0.1.json",
        "hf-results-What is the pap-0.2.json",
        "hf-results-What is the pap-0.3.json"
    ],
    "authors": [
        "hf-results-What is the aut-0.005.json",
        "hf-results-What is the aut-0.01.json",
        "hf-results-What is the aut-0.05.json",
        "hf-results-What is the aut-0.1.json",
        "hf-results-What is the aut-0.2.json",
        "hf-results-What is the aut-0.3.json"
    ],
    "contribution": [
        "hf-results-What is the mai-0.005.json",
        "hf-results-What is the mai-0.01.json",
        "hf-results-What is the mai-0.05.json",
        "hf-results-What is the mai-0.1.json",
        "hf-results-What is the mai-0.2.json",
        "hf-results-What is the mai-0.3.json"
    ]
}

# Loop through each question and its associated files
for question, files in json_file_series.items():
    plt.figure()  # Create a new figure for each question
    durations = []  # List to store all duration lists for violin plot
    labels = []  # List to store labels for each violin plot
    means = []  # List to store the means of duration values

    for json_file in files:
        duration_values = extract_duration_values(base_dir + json_file)
        durations.append(duration_values)
        means.append(np.mean(duration_values))
        # Extract the parameter value from the file name for labeling
        param_value = float(json_file.split(".json")[0].split("-")[-1])
        labels.append(f"{param_value}")

    # Plot the violin plot
    parts = plt.violinplot(durations, showmeans=False, showmedians=False, showextrema=False)

    # Customize the violin plot colors if needed
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Plot mean lines
    for i, mean in enumerate(means):
        plt.plot([i + 1, i + 1], [mean, mean], color='red', linestyle='-', linewidth=2)

    plt.xticks(np.arange(1, len(labels) + 1), labels)

    # Customize the plot
    plt.xlabel('Configuration')
    plt.ylabel('Duration (s)')
    plt.title(f'Violin Plot of Duration for {question}')
    plt.grid(True)

    # Save the plot for each question
    plt.savefig(f'../figure/hf/hf_violin-{question}.pdf')