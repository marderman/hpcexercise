#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from random import sample

def read_data(file_path):
    data = {'TRA': [], 'TRA_TIM': [], 'TST': []}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if parts[0] == 'TRA':
                data['TRA'].append((int(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), float(parts[6])))
            elif parts[0] == 'TRA_TIM':
                data['TRA_TIM'].append((int(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), float(parts[5])))
            elif parts[0] == 'TST':
                data['TST'].append((int(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]), float(parts[6])))

    return data

def plot_loss_curve(data, num_groups_to_plot = -1):
    df_tra = pd.DataFrame(data['TRA'], columns=['Batch_Size', 'Learning_Rate', 'Epoch', 'Total_epochs', 'Minibatch_chunk', 'Loss'])
    grouped_tra = df_tra.groupby(['Batch_Size', 'Learning_Rate', 'Total_epochs'])

    fig, axes = plt.subplots(figsize=(11.69, 8.27), layout='tight')

    line_styles = ['-', '--', '-.', ':']  # Define line styles
    markers = ['o', 's', '^', 'D', 'None']  # Define markers

    filtered_groups = [(name, group) for name, group in grouped_tra if (name[1] == 64 and name[2] == 4)]

    sorted_groups = sorted(filtered_groups, key=lambda x: grouped_tra.get_group(x[0]).first_valid_index())

    if (num_groups_to_plot == -1):
        groups_to_plot = sorted_groups
    else:
        groups_to_plot = sample(list(sorted_groups), min(num_groups_to_plot, len(sorted_groups)))

    # Plotting logic
    for i, (name, group) in enumerate(groups_to_plot):
        axes.plot(
            range(1, len(group) + 1),
            group['Loss'],
            label=f"Batch Size: {name[0]}, LR: {name[1]}, Epochs: {name[2]}",
            linestyle = line_styles[i % len(line_styles)],
            marker = markers[i % len(markers)])

    # Add labels and title
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Loss')
    axes.set_title('Training Loss Curve')
    axes.yaxis.grid(True)

    handles, labels = axes.get_legend_handles_labels()
    handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split(',')[0].split(': ')[1]), float(x[1].split(',')[1].split(': ')[1])))
    handles, labels = zip(*handles_labels)
    axes.legend(handles, labels)

    plt.savefig("figure.png", format='png', bbox_inches='tight', pad_inches=0.1)
    # axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
    plt.show()

def plot_accuracy(data, num_groups_to_plot= -1):
    df_tst = pd.DataFrame(data['TST'], columns=['Batch_Size', 'Learning_Rate', 'Total_epochs', 'TT', 'Accuracy', 'Time'])
    grouped_tst = df_tst.groupby(['Batch_Size', 'Learning_Rate'])

    fig, ax = plt.subplots(figsize=(11.69, 8.27), layout = 'tight')  # A4 size in inches

    line_styles = ['-', '--', '-.', ':']  # Define line styles
    markers = ['o', 's', '^', 'D', 'None']  # Define markers

    filtered_groups = [(name, group) for name, group in grouped_tst if (name[1] > 0.00001)]

    if (num_groups_to_plot == -1):
        groups_to_plot = filtered_groups
    else:
        groups_to_plot = sample(list(filtered_groups), min(num_groups_to_plot, len(filtered_groups)))

    for i, (name, group) in enumerate(groups_to_plot):
        ax.plot(group['Total_epochs'], group['Accuracy'], label=f"BS: {name[0]}, LR: {name[1]}", linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])

    # Add horizontal grid lines
    ax.yaxis.grid(True)

    # Sort legend items based on the custom sorting key
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = sorted(zip(handles, labels), key=lambda x: (int(x[1].split(',')[0].split(': ')[1]), float(x[1].split(',')[1].split(': ')[1])))
    handles, labels = zip(*handles_labels)
    ax.legend(handles, labels, loc='lower right')

    ax.set_title('Accuracy of the model')
    ax.set_xlabel('Training epochs [-]')
    ax.set_ylabel('Accuracy [%]')

    # Save the figure
    plt.savefig('accuracy_plot.png', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_speedup(data_cpu, data_gpu):
    df_tim_cpu = pd.DataFrame(data_cpu['TRA_TIM'], columns=['Batch_Size', 'Learning_Rate', 'Total_epochs', 'Magic_Value', 'Execution_Time'])
    df_tim_gpu = pd.DataFrame(data_gpu['TRA_TIM'], columns=['Batch_Size', 'Learning_Rate', 'Total_epochs', 'Magic_Value', 'Execution_Time'])

    grouped_tim_cpu = df_tim_cpu.groupby(['Batch_Size', 'Learning_Rate'])
    grouped_tim_gpu = df_tim_gpu.groupby(['Batch_Size', 'Learning_Rate'])

    fig, ax = plt.subplots(figsize=(11.69, 8.27), layout = 'tight')  # A4 size in inches
    fig.suptitle('Speedup Comparison (GPU vs CPU)', y=1.02)

    line_styles = ['-', '--', '-.', ':']  # Define line styles
    markers = ['o', 's', '^', 'D', 'None']  # Define markers

    for i, group_key_cpu in enumerate(grouped_tim_cpu.groups.keys()):
        group_key_gpu = group_key_cpu

        bs, lr_cpu = group_key_cpu
        if not(
                (lr_cpu == 0.0001 and bs == 4)
                or (lr_cpu == 0.001 and bs == 4)
                or (lr_cpu == 0.001 and bs == 10)
                or (lr_cpu == 0.01 and bs == 64)
                or (lr_cpu == 0.01 and bs == 256)
                or (lr_cpu == 0.1 and bs == 256)
        ):
            continue

        execution_time_cpu = grouped_tim_cpu.get_group(group_key_cpu)
        execution_time_gpu = grouped_tim_gpu.get_group(group_key_gpu)

        speedup = execution_time_cpu['Execution_Time'] / execution_time_gpu['Execution_Time']

        ax.plot(
            execution_time_cpu['Total_epochs'],
            speedup,
            label=f"BS {group_key_cpu[0]}, LR {group_key_cpu[1]}",
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)]
        )

    # Add horizontal grid lines
    ax.yaxis.grid(True)

    ax.legend(loc='upper right')  # Place the legend outside to the top-right

    ax.set_xlabel('Total Epochs [-]')
    ax.set_ylabel('Speedup [-]')

    # Save the figure
    plt.savefig('speedup.png', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def main():
    cpu_data = read_data('cpu_train.csv')
    gpu_data = read_data('gpu_train.csv')

    # Plot the loss curve
    plot_loss_curve(cpu_data, -1)

    # plot_accuracy(data)

    # plot_speedup(cpu_data, gpu_data)

if __name__ == "__main__":
    main()
