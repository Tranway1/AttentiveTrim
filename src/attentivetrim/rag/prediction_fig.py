import json
import matplotlib.pyplot as plt

BUCKET_NUM = 9
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_data(data):
    iteration_counts = []
    average_lengths = []
    for result in data['results']:
        if result['iteration_count'] > 20000:
            continue
        iteration_counts.append(result['iteration_count'])
        average_lengths.append(result['top_50']['average_10'])
    return iteration_counts, average_lengths


def plot_data(iteration_counts, average_lengths,model_name, dataset):
    plt.figure(figsize=(10, 6))
    plt.scatter(iteration_counts, average_lengths, color='blue')
    # iterate over the points and check if they are close to the y = x line


    # Calculate the range for the lines
    max_value = max(max(iteration_counts), max(average_lengths))
    min_value = min(min(iteration_counts), min(average_lengths))

    bucket_size = (max_value - min_value) / BUCKET_NUM / 2
    equal_points = 0
    for i in range(len(iteration_counts)):
        if abs(iteration_counts[i] - average_lengths[i]) <= bucket_size:
            equal_points += 1

    print(f"Number of points close to the y = x line: {equal_points}")
    equal_ratio = equal_points / len(iteration_counts)

    # Plot y = x line
    plt.plot([min_value, max_value], [min_value, max_value], 'k-', label='y = x')

    # Plot y = x + 50 line
    plt.plot([min_value, max_value], [min_value + bucket_size, max_value + bucket_size], 'y-', label=f'y = x + {bucket_size:.2f} ({equal_ratio:.2f})')

    # Plot y = x - 50 line
    plt.plot([min_value, max_value], [min_value - bucket_size, max_value - bucket_size], 'y-', label=f'y = x - {bucket_size:.2f} (bucket size: {2* bucket_size:.2f})')

    plt.title('Dot Plot of Iteration Counts vs Average Lengths')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.legend()
    plt.savefig('iteration_counts_vs_average_lengths_'+dataset+'_'+model_name+'.png')


def main():
    model_name = 'uae'
    dataset = 'google-qa'
    # file_path = '/Users/chunwei/research/llm-scheduling/profiling-Meta-Llama-3-8B-Instruct-length-' + model_name + '-top20.json'  # Update this path to your actual JSON file location
    file_path = f'/Users/chunwei/research/llm-scheduling/{dataset}-length-{model_name}-top50.json'


    data = load_data(file_path)
    iteration_counts, average_lengths = extract_data(data)
    plot_data(iteration_counts, average_lengths, model_name, dataset)


if __name__ == "__main__":
    main()