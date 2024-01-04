#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data into a DataFrame
data = pd.read_csv('slurm.out')

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(20/2.54, 28.7/2.54), layout="tight")

# Plot execution time on the left y-axis
for process, group in data.groupby('# Processes'):
    ax1.plot(group['# Elements'], group['Time [s]'], label=f'{process} Processes', linestyle='-', marker='o')

# Add labels and legend for execution time
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_xlabel('Number of Elements [-]')
ax1.set_ylabel('Time [s]')
ax1.set_title('Execution Time depending on problem size')
ax1.legend()

# Calculate speedup relative to 2 processes and plot on the right y-axis
baseline_time = data[data['# Processes'] == 1]['Time [s]'].values
for process, group in data.groupby('# Processes'):
    if (process != 1):
        speedup = baseline_time / group['Time [s]'].values
        ax2.plot(group['# Elements'], speedup, label=f'{process} Processes', linestyle='--', marker='x')

ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xscale('log')
ax2.set_xticks(data['# Elements'].unique())
ax2.set_xticklabels(data['# Elements'].unique())
ax2.minorticks_off()
ax2.set_xlabel('Number of Elements [-]')
ax2.set_ylabel('Speedup [-]')
ax2.set_title('Speedup for different problem size (against 1 process computation)')
ax2.legend()

# Calculate efficiency and plot on the right y-axis
for process, group in data.groupby('# Processes'):
    if (process != 1):
        speedup = baseline_time / group['Time [s]'].values
        efficiency = speedup / int(process)
        ax3.plot(group['# Elements'], efficiency, label=f'{process} Processes', linestyle='-.', marker='^')

ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.set_xscale('log')
ax3.set_xticks(data['# Elements'].unique())
ax3.set_xticklabels(data['# Elements'].unique())
ax3.minorticks_off()
ax3.set_xlabel('Number of Elements [-]')
ax3.set_ylabel('Efficiency [-]')
ax3.set_title('Efficiency depending on problem size (against 1 process computation)')
ax3.legend(ncol = 2, loc="lower right", framealpha=0.5)

plt.savefig("meas_data.png",dpi=400, format="png")
# plt.show()
