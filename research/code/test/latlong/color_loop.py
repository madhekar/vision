import matplotlib.pyplot as plt
import numpy as np

# Get the viridis colormap
viridis_cmap = plt.cm.viridis

# Number of lines to plot
num_lines = 20

# Data for plotting (example)
x = np.linspace(0, 10, 200)

# Loop and plot with incrementing colors
for i in range(num_lines):
    # Calculate a normalized value for color mapping
    normalized_value = i / (num_lines - 1)
    
    # Get the corresponding color from the colormap
    color = viridis_cmap(normalized_value)
    
    # Plot a line with the incremented color
    plt.plot(x, np.sin(x + i), color=color)

plt.show()