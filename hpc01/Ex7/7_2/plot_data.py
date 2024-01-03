#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data into a DataFrame
data = pd.read_csv('your_data.csv')

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 12), sharex=True)

# Plot execution time on the left y-axis
for process, group in data.groupby('# Processes'):
    ax1.plot(group['# Elements'], group['Time [s]'], label=f'{process} Processes', linestyle='-', marker='o')

# Add labels and legend for execution time
ax1.set_ylabel('Time [s]')
ax1.set_title('Execution Time vs. Number of Elements')
ax1.legend()

# Calculate speedup relative to 2 processes and plot on the right y-axis
baseline_time = data[data['# Processes'] == 2]['Time [s]'].values
for process, group in data.groupby('# Processes'):
    speedup = baseline_time / group['Time [s]'].values
    ax2.plot(group['# Elements'], speedup, label=f'Speedup ({process} Processes)', linestyle='--', marker='x')

# Add labels and legend for speedup
ax2.set_ylabel('Speedup')
ax2.set_title('Speedup vs. Number of Elements')
ax2.legend()

# Calculate efficiency and plot on the right y-axis
for process, group in data.groupby('# Processes'):
    efficiency = speedup / int(process)
    ax3.plot(group['# Elements'], efficiency, label=f'Efficiency ({process} Processes)', linestyle='-.', marker='^')

# Add labels and legend for efficiency
ax3.set_xlabel('Number of Elements')
ax3.set_ylabel('Efficiency')
ax3.set_title('Efficiency vs. Number of Elements')
ax3.legend()

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot
plt.show()
