import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import binom


def grid_aprox(grid_points=100, success=6, tosses=9):
    p_grid = np.linspace(0., 1., grid_points)
    prior = np.repeat(1., grid_points)
    likehood = binom.pmf(success, tosses, p_grid)
    posterior = likehood * prior
    posterior = posterior / posterior.sum()

    return p_grid, posterior


# "Robert Fico" (klame|klamal)
count_lied = 47400
# "Robert Fico"
all_mentions = 385000

_, ax = plt.subplots(2, 1)

print('fico klamal')
p_grid, posterior = grid_aprox(1000000, count_lied, all_mentions)
ax[0].plot(p_grid, posterior)
ax[0].set_title('distribucia pravdepodobnosti, ze R.Fico klame')

samples = np.random.choice(p_grid, p=posterior, size=int(1e6), replace=True)
print('high posterior density percentile interval 95:' + str(pm.hpd(samples, alpha=0.95)))
print('mean and median : %f,%f' % (np.mean(samples), np.median(samples)))

print('kalinak klamal')
count_lied = 17900
all_mentions = 454000
p_grid, posterior = grid_aprox(1000000, count_lied, all_mentions)
ax[1].plot(p_grid, posterior)
ax[1].set_title('distribucia pravdepodobnosti, ze R.Kalinak klame')

samples = np.random.choice(p_grid, p=posterior, size=int(1e6), replace=True)
print('high posterior density percentile interval 95:' + str(pm.hpd(samples, alpha=0.95)))
print('mean and median : %f,%f' % (np.mean(samples), np.median(samples)))

plt.show()
