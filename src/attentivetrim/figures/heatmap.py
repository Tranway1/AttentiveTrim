import json
import math

import matplotlib.pyplot as plt
import numpy as np

task="test-baselines"
ranges = []
with open("../data/test_v16_inputfile100-result-What are the ba-location.json", "rb") as file:
    data = file.read()
    json_obj = json.loads(data)

for loc in json_obj["files"]:
    minv = loc["start"]/loc["total_chars"]
    maxv = loc["end"]/loc["total_chars"]
    print("min and max:", minv, maxv)
    ranges.append((minv, maxv))

# Determine the overall min and max from the ranges for the heatmap extent
overall_min = 0.0
overall_max = 1.0

# Create a frequency matrix based on the ranges
resolution = 0.001
index_range =  1/resolution
bins = np.arange(overall_min, overall_max, resolution)
frequency_matrix = np.zeros(len(bins)-1)

# Populate the frequency matrix based on the ranges
for r in ranges:
    start_index = math.floor(float(r[0] - overall_min)/resolution)
    end_index = math.ceil(float(r[1] - overall_min)/resolution)
    frequency_matrix[start_index:end_index] += 1

# Plotting the heatmap
plt.figure(figsize=(5, 2))
extent = [overall_min, overall_max, 0, 1]

# Change the colormap to 'viridis' (you can choose another one like 'inferno', 'magma', 'cividis', etc.)
cmap_choice = "inferno"
heatmap = plt.imshow(frequency_matrix[np.newaxis, :], cmap=cmap_choice, aspect="auto", extent=extent)

plt.xlabel("Character position")
plt.title(json_obj["question"])
plt.yticks([])  # Hide y-axis ticks as they don't represent meaningful data in this context

# Adding a colorbar as a legend
cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.05)
cbar.set_label('Frequency')

plt.tight_layout()
plt.savefig("../data/heatmap-"+task+".pdf")

# draw a line chart with x from 0 to 1 and y as the frequency_matrix
plt.figure(figsize=(5, 2))
plt.plot(bins[:-1], frequency_matrix)
plt.xlabel("Character position")
plt.ylabel("Frequency")
plt.title(json_obj["question"])
plt.tight_layout()
plt.savefig("../data/linechart-"+task+".pdf")

# save the index and the frequency_matrix to a csv file
np.savetxt("../data/frequency-"+task+".csv",
           np.column_stack((bins[:-1] * index_range, frequency_matrix)),
           delimiter=",",
           fmt='%.3f')