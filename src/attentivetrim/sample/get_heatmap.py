import json
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def analyze_and_plot_location_data(json_file_name, resolution=0.001, sample_ratio=0.3, init_seed=0):
    # Seed for reproducibility
    random.seed(init_seed)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    loc_dir = os.path.join(cur_dir, "../../../grd_loc")
    heatmap_dir = os.path.join(cur_dir, "../../../heatmap")

    # Infer task name from file path base name
    task = os.path.basename(json_file_name).split('.')[0]

    json_file_path = os.path.join(loc_dir, json_file_name)

    # Read and parse the JSON file
    with open(json_file_path, "rb") as file:
        data = file.read()
    json_obj = json.loads(data)

    # Choose a sample from the entries
    files = json_obj["files"]
    sample_size = int(len(files) * sample_ratio)
    chosen_files = random.sample(files, sample_size)

    # Save chosen file entries as a list in json
    output_data = {"chosen_files": chosen_files}

    # Extract ranges from the chosen entries
    ranges = []
    for loc in chosen_files:

        if loc["groundtruth"].lower() == "none":
            print(f"Skipping {loc['file']} as it has no groundtruth")
            continue
        minv = loc["start"] / loc["total_chars"]
        maxv = loc["end"] / loc["total_chars"]
        ranges.append((minv, maxv))

    # Create a frequency matrix based on the ranges
    overall_min, overall_max = 0.0, 1.0
    bins = np.arange(overall_min, overall_max, resolution)
    frequency_matrix = np.zeros(len(bins) - 1)

    # Populate the frequency matrix based on the ranges
    for r in ranges:
        start_index = math.floor((r[0] - overall_min) / resolution)
        end_index = math.ceil((r[1] - overall_min) / resolution)
        frequency_matrix[start_index:end_index] += 1

    # Append the heatmap data to JSON object
    output_data["heatmap"] = frequency_matrix.tolist()

    # Plotting the heatmap
    plt.figure(figsize=(5, 2))
    extent = [overall_min, overall_max, 0, 1]
    cmap_choice = "inferno"
    heatmap = plt.imshow(frequency_matrix[np.newaxis, :], cmap=cmap_choice, aspect="auto", extent=extent)
    plt.xlabel("Character Position")
    plt.title(json_obj["question"])
    plt.yticks([])
    cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.05)
    cbar.set_label('Frequency')
    plt.tight_layout()
    heatmap_fig_path = os.path.join(heatmap_dir, f"heatmap-{task}.pdf")
    heatmap_json_path = os.path.join(heatmap_dir, f"heatmap-{task}.json")
    plt.savefig(heatmap_fig_path)

    # Write JSON with chosen files and heatmap data
    with open(heatmap_json_path, "w") as outfile:
        json.dump(output_data, outfile, indent=4)



if __name__ == "__main__":
    analyze_and_plot_location_data("notice_What is the date of the notice?-location.json")