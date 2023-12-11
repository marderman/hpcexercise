import matplotlib.pyplot as plt
import numpy as np

X = ['With Profiling','Without profiling']

YProfiling = [5778.26,0.234846]
ZWithoutProfiling = [ 27.5829,4.48167 ]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, YProfiling, 0.4, label='Thrust')
plt.bar(X_axis + 0.2, ZWithoutProfiling, 0.4, label='Naive')


plt.yscale('log') 
plt.xticks(X_axis, X) 
plt.xlabel("Implementation") 
plt.ylabel("Reduction Execution Time in ms") 
plt.title("Execution Time with and without Profiling") 
plt.legend() 
plt.savefig('profiling.png')
