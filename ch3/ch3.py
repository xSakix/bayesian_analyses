import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import mode

import seaborn as sns

p_positive_when_vampire = 0.95
p_positive_when_mortal = 0.01
p_is_vampire = 0.001
p_positive = p_positive_when_vampire*p_is_vampire+p_positive_when_mortal*(1. - p_is_vampire)

p_vampire_positive = p_positive_when_vampire*p_is_vampire / p_positive

print('the probability that you are a vampire given the vampire blood test was positive:%f'%p_vampire_positive)

def grid_aprox(grid_points=100,success=6,tosses=9):
    p_grid = np.linspace(0.,1.,grid_points)
    prior = np.repeat(5,grid_points)
    likehood = binom.pmf(success,tosses,p_grid)
    posterior = likehood*prior
    posterior = posterior/posterior.sum()
    
    return p_grid, posterior

print('--------------6/9-----------')
p_grid,posterior = grid_aprox(1000,6,9)

# plt.plot(p_grid,posterior)
# plt.show()

samples = np.random.choice(p_grid,p=posterior,size=int(1e4),replace=True)
_,(ax0,ax1,ax2) = plt.subplots(1,3)
ax0.plot(p_grid,posterior)
ax1.plot(samples,'o')
sns.kdeplot(samples,ax=ax2)
plt.show()

p_less_then_0_5 = sum(posterior[p_grid < 0.5])
print('posterior < 0.5 = %f'%p_less_then_0_5)
samples_less_0_5 = sum(samples < 0.5)/1e4
print('samples < 0.5 = %f'%samples_less_0_5)

print('between 0.5 and 0.75 posterior:%f'%(sum(posterior[(p_grid > 0.5) & (p_grid < 0.75)])))
print('between 0.5 and 0.75 samples:%f'%(sum((samples > 0.5) & (samples < 0.75))/1e4))

print('80 percentile:'+str(np.percentile(samples,80)))
print('middle 80 percentile(between 10 and 90 percent):'+str(np.percentile(samples,[10,90])))
print('middle 50 percentile(between 25 and 75 percent):'+str(np.percentile(samples,[25,75])))
print('high posterior density percentile interval 50:'+str(pm.hpd(samples,alpha=0.5)))
print('middle 50 percentile(between 80 and 90 percent):'+str(np.percentile(samples,[80,90])))
print('high posterior density percentile interval 80:'+str(pm.hpd(samples,alpha=0.8)))
print('high posterior density percentile interval 90:'+str(pm.hpd(samples,alpha=0.9)))
print('high posterior density percentile interval 95:'+str(pm.hpd(samples,alpha=0.95)))
print('maximum posteriori at prob =(%f,%f)'%(max(posterior),p_grid[posterior == max(posterior)]))
print('maximum posteriori at prob from samples: %f'%mode(samples)[0])
print('mean and median : %f,%f' %(np.mean(samples),np.median(samples)))
print('expected loss:%f'%(sum(posterior*abs(0.5-p_grid))))
loss = [sum(posterior*abs(p-p_grid)) for p in p_grid]
plt.plot(p_grid,loss)
plt.show()
print('min loss at probability:%f'%p_grid[loss == min(loss)])

print('expected loss:%f'%(sum(posterior*np.power(0.5-p_grid,2))))
loss = [sum(posterior*np.power(p-p_grid,2)) for p in p_grid]
plt.plot(p_grid,loss)
plt.show()
print('min loss at probability:%f'%p_grid[loss == min(loss)])

print('--------------3/3-----------')
p_grid,posterior = grid_aprox(1000,3,3)
samples = np.random.choice(p_grid,p=posterior,size=int(1e4),replace=True)
_,(ax0,ax1,ax2) = plt.subplots(1,3)
ax0.plot(p_grid,posterior)
ax1.plot(samples,'o')
sns.kdeplot(samples,ax=ax2)
plt.show()
print('middle 50 percentile(between 25 and 75 percent):'+str(np.percentile(samples,[25,75])))
print('high posterior density percentile interval 50:'+str(pm.hpd(samples,alpha=0.5)))
print('middle 50 percentile(between 80 and 90 percent):'+str(np.percentile(samples,[80,90])))
print('high posterior density percentile interval 80:'+str(pm.hpd(samples,alpha=0.8)))
print('high posterior density percentile interval 90:'+str(pm.hpd(samples,alpha=0.9)))
print('high posterior density percentile interval 95:'+str(pm.hpd(samples,alpha=0.95)))
print('maximum posteriori at prob =(%f,%f)'%(max(posterior),p_grid[posterior == max(posterior)]))
print('maximum posteriori at prob from samples: %f'%mode(samples)[0])
print('mean and median : %f,%f' %(np.mean(samples),np.median(samples)))

print('expected loss:%f'%(sum(posterior*abs(0.5-p_grid))))
loss = [sum(posterior*abs(p-p_grid)) for p in p_grid]
plt.plot(p_grid,loss)
plt.show()
print('min loss at probability:%f'%p_grid[loss == min(loss)])

print('expected loss:%f'%(sum(posterior*np.power(0.5-p_grid,2))))
loss = [sum(posterior*np.power(p-p_grid,2)) for p in p_grid]
plt.plot(p_grid,loss)
plt.show()
print('min loss at probability:%f'%p_grid[loss == min(loss)])


print('-----------------simulating---------')
# simulate globe tossing for water with 0.7 prob. in 2 throws
data = binom.pmf(range(3),2,0.7)
# 0 water 0.09, 1 water 0.42, 2 water 0.49
print(data)
# sample from distribution of simulation data - it is binom, so sample from binom distribution
# we are generating observations...that mean how many times did we see water when we tossed the globe 2x
samples = binom.rvs(n=2, p=0.7, size=1)
print(samples)
samples = binom.rvs(n=2, p=0.7, size=10)
print(samples)
dummy_w = binom.rvs(n=2, p=0.7, size=100000)
means = [(dummy_w == i).mean() for i in range(3)]
print(means)

dummy_w = binom.rvs(n=9, p=0.7, size=100000)
means = [(dummy_w == i).mean() for i in range(9)]
print(means)
plt.hist(dummy_w,bins=50)
plt.show()

dummy_w = binom.rvs(n=9, p=0.6, size=100000)
means = [(dummy_w == i).mean() for i in range(9)]
print(means)
plt.hist(dummy_w,bins=50)
plt.show()

#generate samples from posterior
p_grid, posterior = grid_aprox(grid_points=1000, success=6, tosses=9)
np.random.seed(100)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
dummy_w = binom.rvs(n=9, p=samples)
_,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(posterior)
ax1.hist(dummy_w,bins=50)
plt.show()










    

