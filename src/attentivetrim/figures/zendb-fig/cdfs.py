import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load configuration
config_path = '../../../../questions/question.json'
with open(config_path) as f:
    data = json.load(f)

dataset = "notice"
questions = data["query"][f'{dataset.upper()}_QUESTIONS']
acc_gpt_dir = "../../../../acc/rouge"
acc_rag_dir = "../../../../acc_rag/rouge"

# Define budgets for each series
# [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
rag_budgets = {
    "paper": {"tr": data["budgets"], "rag": [0.01, 0.04, 0.08, 0.17, 0.25], "trrag": [0.01, 0.04, 0.08, 0.17, 0.24]},
    "notice": {"tr": data["budgets"], "rag": [0.03, 0.17, 0.25, 0.35, 0.66, 0.83],
               "trrag": [0.03, 0.17, 0.24, 0.31, 0.39, 0.41]}
}
budgets = rag_budgets[dataset]

# Filter budgets to include only specific close values
filtered_budgets_dict = {"paper":[0.01, 0.04, 0.05, 0.1, 0.18, 0.2, 0.24, 0.25],
                    "notice":[0.02, 0.03, 0.17, 0.2, 0.24, 0.25, 0.35, 0.4, 0.41, 0.66]
                    }

# Ensure the plot directory exists
plot_dir = f"../../../../plot/cdf/{dataset}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


# Function to parse a JSON file and extract specified metric values
def extract_metric_values(json_file, metric):
    with open(json_file, 'r') as file:
        data = json.load(file)
    metric_values = [item[metric] for item in data['files']]
    return metric_values


# Function to plot the CDF
def plot_cdf(data, label, linestyle):
    sorted_data = np.sort(data)
    cdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    sorted_data = np.append(sorted_data, sorted_data[-1])
    cdf = np.append(cdf, 0)
    plt.plot(sorted_data, cdf, label=label, linestyle=linestyle)


# Define line styles for each series
line_styles = {
    'tr': 'solid',  # Solid line for 'tr'
    'trrag': 'dashed',  # Dashed line for 'trrag'
    'rag': 'dashdot'  # Dash-dot line for 'rag'
}

# Metrics to plot
metrics = ["match", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

# Loop through each question and its associated files
for question in questions:
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for series in ['tr', 'trrag', 'rag']:
            for budget in budgets[series]:
                if budget in filtered_budgets_dict[dataset]:
                    if series == 'tr':
                        acc_file = os.path.join(acc_gpt_dir, f"acc-{budget}-{dataset}_{question}-location-acc.json")
                    else:
                        acc_file = os.path.join(acc_rag_dir,
                                                f"ragresults-{budget}-{dataset}_{question}-{series}-acc.json")

                    if os.path.exists(acc_file):
                        metric_values = extract_metric_values(acc_file, metric)
                        series_label = f"{series} {budget} {metric}"
                        plot_cdf(metric_values, label=series_label, linestyle=line_styles[series])

        plt.xlabel('Similarity Score')
        plt.ylabel('CDF')
        plt.title(f'CDF for {question} - {metric}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlim(1.01, 0.0)  # Reverse the x-axis
        plt.savefig(os.path.join(plot_dir, f"cdf-{dataset}-{question}-{metric}.pdf"))
        plt.close()