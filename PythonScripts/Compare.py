import matplotlib.pyplot as plt
import numpy as np
import csv

data = np.loadtxt('position_data.csv', delimiter=',')
data_optimized = np.loadtxt('optimized_data.csv', delimiter=',')
estimated_pos = data[:, 9:12]  
ground_pos = data[:, 12:]     

optimized_pos = data_optimized[:,9:]

plt.figure(figsize=(20,15))
plt.plot(estimated_pos[:, 0], estimated_pos[:, 2], lw=3, label='Estimated', color='b')
plt.plot(ground_pos[:, 0], ground_pos[:, 2], lw=3, label='Ground Truth', color='r')
plt.plot(optimized_pos[:, 0], optimized_pos[:, 2], lw=3, label='Optimized', color='k')
# Set axis labels
plt.xlabel('X Position')
plt.ylabel('Z Position')

# Add a legend
plt.legend()
plt.grid()

# Display the plot
plt.show()