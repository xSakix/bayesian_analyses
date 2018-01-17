import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

growth = np.random.uniform(1., 1.1, (12, 1000)).prod(0)
# plt.figure('growth from 1. to 1.1 prod')
pm.kdeplot(growth)
plt.show()

growth = np.random.uniform(1., 1.5, (12, 1000)).prod(0)
# plt.figure('growth from 1.1 to 1.5 big step prod')
pm.kdeplot(growth)
plt.show()

big = np.random.uniform(1., 1.01, (12, 1000)).prod(0)
small = np.random.uniform(1., 1.5, (12, 1000)).prod(0)

# plt.figure('small and big in comparision')
_,ax = plt.subplots(1,2)
pm.kdeplot(big, ax=ax[0])
pm.kdeplot(small, ax=ax[1])
plt.show()

log_res = np.log(big)
pm.kdeplot(log_res)
plt.show()