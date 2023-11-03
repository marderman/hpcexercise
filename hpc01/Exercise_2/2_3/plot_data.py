#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import csv

meas_iterations = 10

def process_file(lines):
    count = 0
    thr_sum = 0
    szs = []
    thrp = []

    for line in lines:
        parts = line.split()

        if len(parts) >= 7:
            thr_sum += float(parts[7])
            count +=1

            if (count == meas_iterations):
                szs.append(int(parts[2]))
                thrp.append(thr_sum/meas_iterations)
                thr_sum = 0
                count = 0
    return szs, thrp

if __name__ == "__main__":

    one_clus_block_szs = []
    one_clus_block_thrp = []
    two_clus_block_szs = []
    two_clus_block_thrp = []
    one_clus_nblock_szs = []
    one_clus_nblock_thrp = []
    two_clus_nblock_szs = []
    two_clus_nblock_thrp = []

    # -----------------------------------------------------------------------------------------
    # Read input file
    # -----------------------------------------------------------------------------------------
    with open('1clus_block.txt', newline='\n') as f_data:
        lines = f_data.strip().split('\n')
        one_clus_block_szs,one_clus_block_thrp = process_file(lines)

    with open('2clus_block.txt', newline='\n') as f_data:
        lines = f_data.strip().split('\n')
        two_clus_block_szs,two_clus_block_thrp = process_file(lines)

    with open('1clus_nblock.txt', newline='\n') as f_data:
        lines = f_data.strip().split('\n')
        one_clus_nblock_szs,one_clus_nblock_thrp = process_file(lines)

    with open('2clus_nblock.txt', newline='\n') as f_data:
        lines = f_data.strip().split('\n')
        two_clus_nblock_szs,two_clus_nblock_thrp = process_file(lines)


    for i in range(0,len(one_clus_block_szs) -1):
        print("{}, {}, {}, {}".format(one_clus_block_szs[i], one_clus_block_thrp[i]))
    for i in range(0,len(two_clus_block_szs) -1):
        print("{}, {}, {}, {}".format(two_clus_block_szs[i], two_clus_block_thrp[i]))

    for i in range(0,len(one_clus_nblock_szs) -1):
        print("{}, {}, {}, {}".format(one_clus_nblock_szs[i], one_clus_nblock_thrp[i]))
    for i in range(0,len(two_clus_nblock_szs) -1):
        print("{}, {}, {}, {}".format(two_clus_nblock_szs[i], two_clus_nblock_thrp[i]))

    fig, ax = plt.subplots()

    ax.plot(packet_sizes, throughputs, marker='o', linestyle='-', color='b', label='Throughput')

    # plt.xscale('log')  # Set X-axis to a logarithmic scale
    # plt.yscale('log')  # Set Y-axis to a logarithmic scale
    ax.grid(True)
    ax.set_xlabel("Packet size [B]")
    ax.set_ylabel("Througphut [MBps]")

    ax.legend()

    plt.show()
