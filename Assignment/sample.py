import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 1, 3, 6, 7])

cluster = np.array([1, 1, 1, 2, 2])

fig, ax = plt.subplots()

ax.scatter(x[cluster == 1], y[cluster == 1], marker='^')
ax.scatter(x[cluster == 2], y[cluster == 2], marker='s')

plt.savefig('results/plots/' +
            'sample.png', bbox_inches='tight')
plt.show()
