import numpy as np
from scipy.stats import binom
from scipy.stats import uniform
import matplotlib.pyplot as plt

w = 6
n = 9
p_grid = np.linspace(0.,1.,100)
posterior = binom.pmf(w,n,p_grid)*uniform.pdf(p_grid,0.,1.)
posterior = posterior / posterior.sum()

plt.plot(p_grid,posterior)
plt.show()