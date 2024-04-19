import matplotlib.pyplot as plt

# Data
stages = ['loading model', 'tokenizing', 'sending_data', 'generate_res(inference)', 'decoding_res']
durations = [4.88, 0.34, 0.00, 4.18, 0.01]

# Creating the stacked bar chart
fig, ax = plt.subplots()
ax.bar(stages, durations, label='Duration')

# Adding some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
ax.set_title('Time by process stage and overhead')
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, rotation=45, ha="right")
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()