import json
import matplotlib.pyplot as plt
import os

# Load configuration
config_path = '../../../../questions/question.json'
with open(config_path) as f:
    data = json.load(f)

dataset = "notice"
# dataset = "paper"
file_list = data["datasets"][dataset]["list"]
questions = data["query"][f'{dataset.upper()}_QUESTIONS']
acc_gpt_dir = "../../../../acc/gpt"
acc_rag_dir = "../../../../acc_rag/gpt"

# Define budgets for each series
rag_budgets = {
    "paper": {"tr": data["budgets"], "rag": [0.01, 0.04, 0.08, 0.17, 0.25], "trrag": [0.01, 0.04, 0.08, 0.17, 0.24]},
    "notice": {"tr": data["budgets"], "rag": [0.03, 0.17, 0.25, 0.35, 0.66, 0.83], "trrag": [0.03, 0.17, 0.24, 0.31, 0.39, 0.41]}
}
budgets = rag_budgets[dataset]

# Ensure the plot directory exists
plot_dir = f"../../../../plot/{dataset}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


# Function to extract matched ratio from a JSON file
def extract_matched_ratio(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    total_matches = data['total_matches']
    total_files = data['total_files']
    matched_ratio = total_matches / total_files
    return matched_ratio


# Plotting function for each question
def plot_question_accuracy(question):
    plt.figure(figsize=(10, 6))
    budget_map = {
        "What are the state abbreviation and ZIP code of the company?": [0.01, 0.02, 0.05, 0.10, 0.23, 0.35],
        "What are the violation items?": [0.01, 0.04, 0.09, 0.21, 0.34],
        "What is the civil penalty amount suggested for the probable violation?": [0.01, 0.04, 0.10, 0.21, 0.34],
        "What is the date of the notice?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.35],
        "What is the full title of the sender in the notice?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.35],
        "What is the name of the company?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.34],
        "What is the type of violation item?": [0.01, 0.04, 0.09, 0.21, 0.34],
        "What statutory authority is used to propose the Compliance Order?": [0.01, 0.02, 0.05, 0.11, 0.22, 0.35]
    }


    series_labels = ['tr', 'trrag', 'rag', 'att']
    colors = {'tr': 'blue', 'rag': 'green', 'trrag': 'red', 'att': 'orange'}

    for series in series_labels:
        x_values = []
        y_values = []

        if series == 'att':
            cur_budgets = budget_map[question]
        else:
            cur_budgets = budgets[series]

        for budget in cur_budgets:
            if series == 'tr':
                acc_file = os.path.join(acc_gpt_dir, f"acc-{budget}-{dataset}_{question}-location-acc.json")
            elif series == 'att':
                acc_file = os.path.join(acc_gpt_dir, f"accAtt-{budget:.2f}-{dataset}_{question}-location-acc.json")
            else:
                acc_file = os.path.join(acc_rag_dir, f"ragresults-{budget}-{dataset}_{question}-{series}-acc.json")

            if os.path.exists(acc_file):
                matched_ratio = extract_matched_ratio(acc_file)
                x_values.append(budget)
                y_values.append(matched_ratio)

        plt.plot(x_values, y_values, marker='o', linestyle='-', label=f"{series}", color=colors[series])

    plt.xlabel('Budget')
    plt.ylabel('Matched Ratio')
    plt.title(f'Accuracy for {question}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{dataset}_{question}.pdf"))
    plt.close()


# Generate plots for each question
for question in questions:
    plot_question_accuracy(question)