import json
import numpy as np
import matplotlib.pyplot as plt
import os


# Function to parse a JSON file and extract the "match" values
def extract_match_values(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        match_values = [item['match'] for item in data['files']]
        # if value > 1, set it to 1
        match_values = [min(val, 1.0) for val in match_values]
    return match_values


# Function to plot the CDF
def plot_cdf(data, label):
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    # Calculate the CDF values
    cdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Append the maximum x value to the sorted data and a CDF value of 0
    sorted_data = np.append(sorted_data, sorted_data[-1])
    cdf = np.append(cdf, 0)  # Assuming you want the CDF to go down to 0 at the end

    # Plot the CDF
    plt.plot(sorted_data, cdf, label=label)


base_dir = "../data/local-full/"

# Define the mapping of questions to file arrays
json_file_series = {
    "title": [
        "results-What is the pap-0.001-acc-local-0.1.json",
        "results-What is the pap-0.005-acc-local-0.1.json",
        "results-What is the pap-0.05-acc-local-0.1.json",
        "results-What is the pap-0.3-acc-local-0.1.json"
    ],
    "authors": [
        "results-What is the aut-0.005-acc-local-0.1.json",
        "results-What is the aut-0.01-acc-local-0.1.json",
        "results-What is the aut-0.05-acc-local-0.1.json",
        "results-What is the aut-0.1-acc-local-0.1.json"
    ],
    "contribution": [
        "results-What is the mai-0.05-acc-local-full.json",
        "results-What is the mai-0.1-acc-local-full.json",
        "results-What is the mai-0.15-acc-local-full.json",
        "results-What is the mai-0.2-acc-local-full.json",
        "results-What is the mai-0.4-acc-local-full.json",
        "results-What is the mai-0.9-acc-local-full.json"
    ]
}

# Loop through each question and its associated files
for question, files in json_file_series.items():
    plt.figure()  # Create a new figure for each question
    for json_file in files:
        match_values = extract_match_values(base_dir + json_file)
        # Extract the parameter value from the file name
        param_value = float(json_file.split("-acc")[0].split("-")[-1])
        series_label = f"{question} {param_value}"
        plot_cdf(match_values, label=series_label)

    # Customize the plot
    plt.xlabel('Similarity Score')
    plt.ylabel('CDF')
    plt.title(f'CDF for {question}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(1.01, 0.0)  # Set the x-axis limits to reverse the axis

    # Save the plot for each question
    plt.savefig(f'../figures/figure/cdf-{question}.pdf')