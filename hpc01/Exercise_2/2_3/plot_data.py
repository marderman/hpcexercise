#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import csv

meas_iterations = 10
debug = True

def process_file(lines):
    count = 0
    thr_sum = 0
    szs = []
    thrp = []

    for line in lines[2:]:
        parts = line.split()
        if debug:
            print("Num of fields in line: ", len(parts))

        thr_sum += float(parts[5])
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
    with open('1clus_block.txt', "r", newline='\n') as f_data:
        lines = f_data.readlines()
        one_clus_block_szs,one_clus_block_thrp = process_file(lines)

    with open('2clus_block.txt', "r", newline='\n') as f_data:
        lines = f_data.readlines()
        two_clus_block_szs,two_clus_block_thrp = process_file(lines)

    with open('1clus_nblock.txt', "r", newline='\n') as f_data:
        lines = f_data.readlines()
        one_clus_nblock_szs,one_clus_nblock_thrp = process_file(lines)

    with open('2clus_nblock.txt', "r", newline='\n') as f_data:
        lines = f_data.readlines()
        two_clus_nblock_szs,two_clus_nblock_thrp = process_file(lines)


    if debug:
        for i in range(0,len(one_clus_block_szs)):
            print("{}, {}, {}".format(one_clus_block_szs[i], one_clus_block_thrp[i], one_clus_nblock_thrp[i]))
        for i in range(0,len(two_clus_block_szs)):
            print("{}, {}, {}".format(two_clus_block_szs[i], two_clus_block_thrp[i], two_clus_nblock_thrp[i]))
        for i in range(0,len(two_clus_nblock_szs)):
            print("{}, {}, {}, {}".format(two_clus_nblock_szs[i], one_clus_block_thrp[i], two_clus_block_thrp[i], one_clus_block_thrp[i]/two_clus_block_thrp[i]))

    fig, ax = plt.subplots(3,1, layout = "constrained", figsize = (16/2.54,20/2.54), dpi = 200)

    ax[0].plot(one_clus_block_szs, one_clus_block_thrp, marker='x', linestyle='-', color='blue', label='Blocking send')
    ax[0].plot(one_clus_nblock_szs, one_clus_nblock_thrp, marker='x', linestyle='-', color='red', label='Non-blocking send')
    ax[0].set_title("MPI Throughput on a single cluser")
    ax[0].grid(True)
    ax[0].set_xlabel("Message size [B]")
    ax[0].set_ylabel("Througphut [MBps]")
    ax[0].legend()
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(1, 10**7)

    ax[1].plot(two_clus_block_szs, two_clus_block_thrp, marker='x', linestyle='-', color='green', label='Blocking send')
    ax[1].plot(two_clus_nblock_szs, two_clus_nblock_thrp, marker='x', linestyle='-', color='orange', label='Non-blocking send')
    ax[1].set_title("MPI Throughput between two clusters")
    ax[1].grid(True)
    ax[1].set_xlabel("Message size [B]")
    ax[1].set_ylabel("Througphut [MBps]")
    ax[1].legend()
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1, 10**7)

    ax[2].plot(one_clus_block_szs, one_clus_block_thrp, marker='x', linestyle='-', color='blue', label='One cluster')
    ax[2].plot(two_clus_block_szs, two_clus_block_thrp, marker='x', linestyle='-', color='green', label='Two clusters')
    ax[2].set_title("MPI Throughput on one vs. two clusters (blocking send)")
    ax[2].grid(True)
    ax[2].set_xlabel("Message size [B]")
    ax[2].set_ylabel("Througphut [MBps]")
    ax[2].legend()
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlim(1, 10**7)

    plt.savefig("image.png")
    # plt.show()
