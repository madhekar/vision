import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

num_colors = 20
viridis_colors = [plt.cm.viridis(i/num_colors) for i in range(num_colors)]

x = np.linspace(0, 10, 200)
fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', viridis_colors))

for i in range(num_colors):
    y = np.sin(x + i * np.pi / num_colors)
    ax.plot(x, y)

plt.show()