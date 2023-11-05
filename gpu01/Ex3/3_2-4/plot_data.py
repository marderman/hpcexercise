#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('slurm.out')

    # Group the data by 'Size [B]' and create a separate plot for each group
    groups = df.groupby(' Size [B]')

    # Plot each group
    for name, group in groups:
        plt.plot(group[' # of Threads per Block'], group[' Throughput [GBps]'], linestyle='-', marker='x', label=f'{name} [B]')

    # Set plot labels and legend
    plt.xlabel('# of Threads per Block')
    plt.ylabel('Throughput [GBps]')
    plt.legend()

    # Show the plot
    plt.show()
