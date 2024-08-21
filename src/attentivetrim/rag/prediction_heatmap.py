import json
import numpy as np
import matplotlib.pyplot as plt

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

def quantize_data(values, num_buckets, min_val, max_val):
    range_val = max_val - min_val
    bucket_size = range_val / num_buckets
    bucketed_data = np.floor((np.array(values) - min_val) / bucket_size).astype(int)
    bucketed_data[bucketed_data == num_buckets] = num_buckets - 1  # Handle the max value case
    return bucketed_data, min_val, max_val, bucket_size

def create_heatmap(iteration_counts, average_lengths, num_buckets, model_name, dataset):
    min_val, max_val = min(min(iteration_counts), min(average_lengths)), max(max(iteration_counts),max(average_lengths))
    bucketed_iterations, _, _, _ = quantize_data(iteration_counts, num_buckets, min_val, max_val)
    bucketed_lengths, min_length, max_length, _ = quantize_data(average_lengths, num_buckets, min_val, max_val)

    heatmap_matrix = np.zeros((num_buckets, num_buckets))
    for i in range(num_buckets):
        column_points = bucketed_lengths[bucketed_iterations == i]
        if len(column_points) > 0:
            for j in column_points:
                heatmap_matrix[j, i] += 1
            heatmap_matrix[:, i] /= len(column_points)  # Normalize by column totals

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(heatmap_matrix, cmap='viridis', aspect='auto', origin='lower',
                   extent=[min_length, max_length, min_length, max_length])
    plt.colorbar(im, label='Ratio')
    plt.title('Heatmap of Quantized Data')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(False)

    # Calculate the centers of the bins for annotations
    x_centers = np.linspace(min_length, max_length, num_buckets, endpoint=False)
    x_centers += (x_centers[1] - x_centers[0]) / 2
    y_centers = np.linspace(min_length, max_length, num_buckets, endpoint=False)
    y_centers += (y_centers[1] - y_centers[0]) / 2

    # Annotate each cell with the numeric value
    for i in range(num_buckets):
        for j in range(num_buckets):
            text = ax.text(x_centers[j], y_centers[i], f"{heatmap_matrix[i, j]:.2f}",
                           ha="center", va="center", color="w")

    print('Saving heatmap to heatmap_iteration_counts_vs_average_lengths_' + model_name + '.png')
    plt.savefig('heatmap_'+dataset+'_' + model_name + '.png')
def main():
    model_name = 'uae'
    dataset = 'google-qa'
    # file_path = '/Users/chunwei/research/llm-scheduling/profiling-Meta-Llama-3-8B-Instruct-length-' + model_name + '-top20.json'  # Update this path to your actual JSON file location
    file_path = f'/Users/chunwei/research/llm-scheduling/{dataset}-length-{model_name}-top50.json'
    data = load_data(file_path)
    iteration_counts, average_lengths = extract_data(data)
    create_heatmap(iteration_counts, average_lengths, 9, model_name, dataset)

if __name__ == "__main__":
    main()