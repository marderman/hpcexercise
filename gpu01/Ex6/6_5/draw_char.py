#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
data = pd.read_csv('slurm.out')

# Calculate averages for every unique combination of 'Array size', 'Block size', and 'Parallel reduction BW'
data_avg = data.groupby(['Array size', 'Block size']).mean().reset_index()

# Set font size
font_size = 12

# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), tight_layout=True)

# Plot the first chart
for block_size, group in data_avg.groupby('Block size'):
    ax1.plot(group['Array size'], group['Parallel reduction BW'], label=f'Block size {block_size}')

ax1.set_xlabel('Array size [-]', fontsize=font_size)
ax1.set_ylabel('Parallel reduction BW [GBps]', fontsize=font_size)
ax1.legend(title='Legend', loc='upper left', fontsize=font_size)
ax1.set_title('Relation between Array size and Parallel reduction BW', fontsize=font_size)
ax1.grid(True)

ax2_seq = ax2.twinx()
# Plot the second chart
lines_ax2, labels_ax2 = [], []
lines_ax2_seq, labels_ax2_seq = [], []

for block_size, group in data_avg.groupby('Block size'):
    line_ax2 = ax2.plot(group['Array size'], group['Parallel reduction BW'], label=f'Par, BS: {block_size}', linestyle='dashed')
    lines_ax2.extend(line_ax2)
    labels_ax2.extend([line.get_label() for line in line_ax2])

    line_ax2_seq = ax2_seq.plot(group['Array size'], group['Sequential reduction BW'], label=f'Seq, BS: {block_size}', linestyle='solid')
    lines_ax2_seq.extend(line_ax2_seq)
    labels_ax2_seq.extend([line.get_label() for line in line_ax2_seq])

# Combine legends for ax2 and ax2_seq
ax2.legend(lines_ax2 + lines_ax2_seq, labels_ax2 + labels_ax2_seq, title='Legend', loc='upper left', fontsize=font_size)

ax2.set_xlabel('Array size [-]', fontsize=font_size)
ax2.set_ylabel('Parallel reduction BW [GBps]', fontsize=font_size)
ax2_seq.set_ylabel('Sequential reduction BW [GBps]', color='tab:blue', fontsize=font_size)

ax2.set_title('Relation between Parallel and Sequential BW with Array size', fontsize=font_size)
ax2.grid(True)

ax2_seq.tick_params(axis='y', labelcolor='tab:blue', labelsize=font_size)

# Save the figure as a PNG file
plt.savefig('your_plot.png')

# Show the figure
plt.show()
