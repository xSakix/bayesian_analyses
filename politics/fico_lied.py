import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import binom


#count_lied = 208000
count_lied = 319
# all_mentions = 412000 
all_mentions = 4490

def grid_aprox(grid_points=100,success=6,tosses=9):
    p_grid = np.linspace(0.,1.,grid_points)
    prior = np.repeat(1.,grid_points)
    likehood = binom.pmf(success,tosses,p_grid)
    posterior = likehood*prior
    posterior = posterior/posterior.sum()
    
    return p_grid, posterior
    
    
print('fico klamal')
p_grid,posterior = grid_aprox(1000000,count_lied,all_mentions)
plt.plot(p_grid,posterior)
plt.show()

samples = np.random.choice(p_grid,p=posterior,size=int(1e6),replace=True)
print('high posterior density percentile interval 95:'+str(pm.hpd(samples,alpha=0.95)))
print('mean and median : %f,%f' %(np.mean(samples),np.median(samples)))


print('kalinak klamal')
count_lied = 112
all_mentions = 1490
p_grid,posterior = grid_aprox(1000000,count_lied,all_mentions)
plt.plot(p_grid,posterior)
plt.show()

samples = np.random.choice(p_grid,p=posterior,size=int(1e6),replace=True)
print('high posterior density percentile interval 95:'+str(pm.hpd(samples,alpha=0.95)))
print('mean and median : %f,%f' %(np.mean(samples),np.median(samples)))
