import pandas as pd
import matplotlib.pyplot as plt
import sys

import os

filename = sys.argv[1]
# Load the CSV file
df = pd.read_csv(filename)

# Extract matrix sizes from filenames
def extract_size(matrix_name):
    return float(matrix_name.split('_')[1])

df['Size'] = df['Matrix1'].apply(extract_size)

print(df)

# Select only numeric columns for averaging
numeric_cols = ['cuBlas (µs)', 'cuSparse (µs)', 'gpuDense (µs)', 'gpuSparse (µs)', 
                'Blas (µs)', 'cpuSparseParallel (µs)', 'cpuDenseParallel (µs)']
# Group by matrix size (assuming they are paired)
df_grouped = df.groupby('Size')[numeric_cols].mean().sort_index(ascending=False)

# Extract x and y values for plotting
sizes = df_grouped.index
methods = ['cuBlas (µs)', 'cuSparse (µs)', 'gpuDense (µs)', 'gpuSparse (µs)', 'Blas (µs)', 'cpuSparseParallel (µs)', 'cpuDenseParallel (µs)']

colors = {
    'cuBlas (µs)': '#1f77b4',  # Blau
    'cuSparse (µs)': '#ff7f0e',  # Orange
    'gpuDense (µs)': '#2ca02c',  # Grün
    'gpuSparse (µs)': '#d62728',  # Rot
    'Blas (µs)': '#999999',  # grau
    'cpuSparseParallel (µs)': '#8c564b',  # Braun
    'cpuDenseParallel (µs)': '#000000'  # schwarz
}

plt.figure(figsize=(10, 10))

# Plot data for each method mit den definierten Farben
for method in methods:
    plt.plot(sizes, df_grouped[method], marker='o', label=method, color=colors[method])
 
# plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix Size')
plt.ylabel('Time (µs)')

title = ""
if filename.find("total") != -1:
    title = "Total Time"
elif filename.find("raw_multiplication") != -1:
    title = "Raw Multiplication Time"
elif filename.find("overhead") != -1:
    title = "Overhead Time"
plt.title('Performance Comparison: ' + title)
plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)


if not os.path.exists("/".join(filename.split("/")[:-2]) + "/plots/"):
    os.makedirs("/".join(filename.split("/")[:-2]) + "/plots/")
output_filename = "/".join(filename.split("/")[:-2]) + "/plots/" + filename.split("/")[-1].replace(".csv", ".svg")
plt.savefig(output_filename, format="svg", bbox_inches='tight')

output_filename = "/".join(filename.split("/")[:-2]) + "/plots/" + filename.split("/")[-1].replace(".csv", "")
plt.savefig(output_filename, bbox_inches='tight')

# Show the plot
plt.show()