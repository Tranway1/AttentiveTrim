import pandas as pd
import matplotlib.pyplot as plt

# Assuming your CSV data is saved in a file named 'data.csv'
csv_data = """
Question vs Text ratio,0.005,0.01,0.05,0.1,0.2,0.4
Title, 0.64,0.62,0.95,1.25,2.09,3.22
Authors, 0.69,0.69,1.00,1.38,1.78,3.17
Contribution, 0.84,1.62,1.42,3.24,2.57,5.85
"""


# Writing the CSV data to a file for demonstration purposes
# In practice, you would load this from an existing file
with open('data.csv', 'w') as file:
    file.write(csv_data)

# Reading the CSV data
df = pd.read_csv('data.csv')

# Setting the 'Question vs Text ratio' column as the index
df.set_index('Question vs Text ratio', inplace=True)

# Converting column labels to float for a linear x-axis
df.columns = df.columns.astype(float)

# Plotting
for question in df.index:
    plt.plot(df.columns, df.loc[question], marker='o', label=question)

# Setting x-axis to be linear from 0.0 to 0.4
plt.xticks([0.005, 0.01, 0.05, 0.1, 0.2, 0.4])

# Adding labels and title
plt.xlabel('text ratio')
plt.ylabel('Runtime (s)')
plt.title('Runtime by Question vs Text Ratio')
plt.legend()

# Optionally, you can set the xlim to ensure the range covers exactly from 0.0 to 0.4
plt.xlim(0.0, 0.4)

plt.savefig("runtime_question_vs_text_ratio_linear_vllm.pdf")