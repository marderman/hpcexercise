import matplotlib.pyplot as plt

# Initialize empty lists to store data
x = []  # Stride values
y = []  # Bandwidth values

# Open the data file
path = "D:/Master/GPU_Computing/hpcexercise/gpu01/Ex3/3_3/template/data.txt"
path="data.txt"
with open(path, 'r') as file:
    for line in file:
        if line.startswith('Strided'):
            # Split the line by semicolon and extract relevant information
            parts = line.split(';')
            stride_value = int(parts[1])
            bandwidth = float(parts[-2])
            
            # Append data to lists
            x.append(stride_value)
            y.append(bandwidth)

# Create a scatter plot
plt.scatter(x, y, label='strided data', color='b')

# Set labels and title
plt.xlabel('stride')
plt.ylabel('bandwidth (GB/s)')
plt.title('strided data bandwidth')

# Show the plot
plt.legend()
plt.grid(True)
plt.show()
