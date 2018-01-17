import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm


def sim_walks(num_of_steps):
    walks = np.random.uniform(-1., 1, (num_of_steps, 1000))
    pos = walks.sum(0)
    return pos


pos = sim_walks(16)

pm.kdeplot(pos)
plt.show()
plt.hist(pos, bins=50)
plt.show()

pos = sim_walks(4)
pm.kdeplot(pos)
plt.show()

pos = sim_walks(8)
pm.kdeplot(pos)
plt.show()


