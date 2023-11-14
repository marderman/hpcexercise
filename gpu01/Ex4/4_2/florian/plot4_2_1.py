import numpy as np
import matplotlib.pyplot as plt
num = 3
if num == 1:
    file = 'centralized_out4_2_1.txt'
    title = 'Bandwith over Threads write to shared Memory'
elif num == 2:
    file = 'centralized_out4_2_2.txt'
    title = 'Bandwith over Threads write from shared Memory'
elif num == 3:
    file = 'centralized_out4_2_3.txt'
    title = 'bandwith over Threads read shared Memory varying Blocksize'
elif num == 4:
    file = 'centralized_out4_2_4.txt'
    title = 'bandwith over Threads write shared Memory varying Blocksize'
elif num == 5:
    file = 'centralized_out4_2_5.txt'


data = np.genfromtxt(file, delimiter=';', skip_header=1, usecols=(2, 4, 6, 8), dtype=[('size', 'int'), ('gDim', 'int'), ('bDim', 'int'), ('bw', 'float')])
if num == 1 or num == 2:
    # Extract unique sizes
    unique_sizes = np.unique(data['size'])
elif num == 3 or num == 4:
    unique_sizes = np.unique(data['gDim'])

# Plotting
fig, ax = plt.subplots()

for size in unique_sizes:
    if num == 1 or num == 2:
        size_data = data[data['size'] == size]
        bDim_values = size_data['bDim']
        bw_values = size_data['bw']

        ax.plot(bDim_values, bw_values, marker='o', linestyle='-', label=f'Size {size}')
    elif num == 3 or num == 4:
        size_data = data[data['gDim'] == size]
        bDim_values = size_data['bDim']
        bw_values = size_data['bw']

        ax.plot(bDim_values, bw_values, marker='o', linestyle='-', label=f'Size {size}')




ax.set_title(title)
ax.set_xlabel('bDim')
ax.set_ylabel('Bandwidth (GB/s)')
ax.legend()
ax.grid(True)
plt.show()