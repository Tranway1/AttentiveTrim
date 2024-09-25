import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_data(data, k=20):
    iteration_counts = []
    average_lengths = []
    for result in data['results']:
        if result['iteration_count'] > 20000:
            break
        iteration_counts.append(result['iteration_count'])
        average_lengths.append(result['top_50'][f'average_{k}'])
    return iteration_counts, average_lengths

def extract_data_vote(data, bucket_size, min, k=20):
    iteration_counts = []
    voted = []
    for result in data['results']:
        if result['iteration_count'] > 20000:
            break
        iteration_counts.append(result['iteration_count'])
        # create dict to store votes
        votes = {}

        # iter from 0 to k
        for i in range(k):
            cand = result['top_50']["records"][i]
            pred = cand['iteration_count']
            quantize = np.floor((pred - min) / bucket_size).astype(int)
            votes[quantize] = votes.get(quantize, 0) + 1
        voted.append(max(votes, key=votes.get)*bucket_size + min)


    return iteration_counts, voted

def quantize_data(values, num_buckets, min_val, max_val):
    range_val = max_val - min_val
    bucket_size = range_val / num_buckets
    bucketed_data = np.floor((np.array(values) - min_val) / bucket_size).astype(int)
    bucketed_data[bucketed_data == num_buckets] = num_buckets - 1  # Handle the max value case
    return bucketed_data, min_val, max_val, bucket_size


def create_heatmap(iteration_counts, average_lengths, num_buckets, model_name, dataset, vote=False, k=0):
    min_val, max_val = min(min(iteration_counts), min(average_lengths)), max(max(iteration_counts),
                                                                             max(average_lengths))
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
    im = ax.imshow(heatmap_matrix, cmap='viridis', aspect='auto', origin='upper',
                   extent=[min_length, max_length, max_length, min_length])
    plt.colorbar(im, label='Ratio')
    plt.title('Heatmap of Quantized Data')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(False)

    # Calculate the centers of the bins for annotations
    x_centers = np.linspace(min_length, max_length, num_buckets, endpoint=False)
    x_centers += (x_centers[1] - x_centers[0]) / 2
    y_centers = np.linspace(max_length, min_length, num_buckets, endpoint=False)
    y_centers += (y_centers[1] - y_centers[0]) / 2

    # Annotate each cell with the numeric value
    for i in range(num_buckets):
        for j in range(num_buckets):
            # Adjust the index for y_centers since the y-axis is reversed
            text = ax.text(x_centers[j], y_centers[num_buckets - 1 - i], f"{heatmap_matrix[i, j]:.2f}", ha="center",
                           va="center", color="w")
    kstr = f'top-{k}' if k > 0 else ''
    if vote:
        print('Saving heatmap to heatmap_iteration_counts_vs_vote_lengths_' + model_name + '_vote.png')
        plt.savefig('figs/heatmap_' + dataset + '_' + model_name + '_vote'+kstr+'.png')
    else:
        print('Saving heatmap to heatmap_iteration_counts_vs_average_lengths_' + model_name + '.png')
        plt.savefig('figs/heatmap_' + dataset + '_' + model_name + kstr+'.png')


def main():
    model_names = ['uae', 'sfr']
    datasets = ['profiling-Meta-Llama-3-8B-Instruct', 'alpaca', 'google-qa']
    top_ks = [5, 10, 20, 30, 40]
    num_buckets = 9
    for model_name in model_names:
        for dataset in datasets:
            for top_k in top_ks:

                file_path = f'/Users/chunwei/research/llm-scheduling/{dataset}-length-{model_name}-top50.json'
                data = load_data(file_path)
                iteration_counts, average_lengths = extract_data(data, k = top_k)
                create_heatmap(iteration_counts, average_lengths, num_buckets, model_name, dataset, k=top_k)

                # vote instead of average over top k
                min_val, max_val = min(min(iteration_counts), min(average_lengths)), max(max(iteration_counts),
                                                                                 max(average_lengths))
                bucket_size = (max_val - min_val) / num_buckets
                iteration_counts, vote_lengths = extract_data_vote(data, bucket_size, min_val, k = top_k)
                create_heatmap(iteration_counts, vote_lengths, num_buckets, model_name, dataset, vote=True, k=top_k)

if __name__ == "__main__":
    main()