#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import csv


if __name__ == "__main__":

    meas_iterations = 10
    labels=[]

    nb_list = []
    tpb_list = []
    st_data=[]
    ast_data=[]

    # -----------------------------------------------------------------------------------------
    # Read input file
    # -----------------------------------------------------------------------------------------
    with open('ex2_out.txt', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        labels = next(spamreader)

        count = 0
        st_total = 0.0
        ast_total = 0.0

        for row in spamreader:

            st = float(row[2])
            ast = float(row[3])
            # Accumulate values from the third column
            st_total += st
            ast_total += ast
            count += 1

            # Check if 10 lines have been processed
            if count == meas_iterations:
                nb = int(row[0])
                tpb = int(row[1])
                st_average = st_total / meas_iterations
                ast_average = ast_total / meas_iterations

                nb_list.append(nb)
                tpb_list.append(tpb)
                st_data.append(st_average)
                ast_data.append(ast_average)

                count = 0
                st_total = 0.0
                ast_total = 0.0

    # for i in range(0,len(st_data) -1):
    #     print("{}, {}, {}, {}".format(nb_list[i], tpb_list[i], st_data[i], ast_data[i]))

    print("Sync time max: ", max(st_data))
    print("Sync time min: ", min(st_data))
    print("Async time max: ", max(ast_data))
    print("Async time min: ", min(ast_data))

    plt.style.use('_mpl-gallery')

    # Plot
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}
    )
    ax.scatter(nb_list, tpb_list, st_data, c = "blue", label = labels[2])
    ax.scatter(nb_list, tpb_list, ast_data, c = "orange", label = labels[3])

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel('Exec time [us]')


    ax.legend()
    # plt.savefig("meas_data.png")

    plt.show()
