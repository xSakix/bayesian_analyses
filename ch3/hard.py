import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import mode
import seaborn as sns
import collections

birth1 = np.array(
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0,
     1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
birth2 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                   1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1,
                   0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

boys = np.sum(birth1) + np.sum(birth2)

g1 = collections.Counter(birth1)
g2 = collections.Counter(birth2)

girls = g1[0]+g2[0]
print('overall boys:' + str(boys))
print('overall girls:' + str(girls))

size = len(birth1)+len(birth2)

priors = np.random.uniform(0.,1.,size)
p_grid = np.linspace(0.,1.,size)
likehood = binom.pmf(boys,size, p=p_grid)
posterior = likehood*priors
posterior = posterior/posterior.sum()
plt.plot(p_grid,posterior)
plt.show()

print('maximum posteriori %f at prob = %f '%(max(posterior),p_grid[posterior == max(posterior)]))


# likehood = binom.pmf(np.sum(birth2),len(birth2), p=p_grid)
# posterior = likehood*posterior
# posterior = posterior/posterior.sum()
#
# plt.plot(p_grid,posterior)
# plt.show()
sample_size=int(10000)
samples = np.random.choice(a=p_grid,p=posterior,size=sample_size)
sns.kdeplot(samples)
plt.show()

print('50% hpd: '+str(pm.hpd(samples,alpha=0.5)))
print('89% hpd: '+str(pm.hpd(samples,alpha=0.89)))
print('97% hpd: '+str(pm.hpd(samples,alpha=0.97)))

#model fits well, max is around 111/200 as is the observation
dummy_w = binom.rvs(n=200,p=samples,size=sample_size)
_,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(posterior)
ax1.hist(dummy_w,bins=50)
plt.show()
means = [(dummy_w == i).mean() for i in range(200)]

print('boys = %f from number of births %f '%(boys,size))
plt.plot(means)
plt.show()

dummy_w = binom.rvs(n=100,p=samples,size=sample_size)
_,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(posterior)
ax1.hist(dummy_w,bins=50)
plt.show()
means = [(dummy_w == i).mean() for i in range(100)]

print('boys = %f from number of births %f '%(np.sum(birth1),len(birth1)))
plt.plot(means)
plt.show()

#simulating boys(second born) folowing girls born first
dummy_w = binom.rvs(n=g1[0],p=samples,size=sample_size)
_,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(posterior)
ax1.hist(dummy_w,bins=50)
plt.show()
means = [(dummy_w == i).mean() for i in range(200)]

print('boys = %f from number of births %f '%(boys,size))
plt.plot(means)
plt.show()




#simulating 51 boys from 1st birth
size = len(birth1)
priors = np.random.uniform(0.,1.,size)
p_grid = np.linspace(0.,1.,size)
likehood = binom.pmf(np.sum(birth1),size, p=p_grid)
posterior = likehood*priors
posterior = posterior/posterior.sum()
plt.plot(p_grid,posterior)
plt.show()

sample_size=int(10000)
samples = np.random.choice(a=p_grid,p=posterior,size=sample_size)
sns.kdeplot(samples)
plt.show()

print('50% hpd: '+str(pm.hpd(samples,alpha=0.5)))
print('89% hpd: '+str(pm.hpd(samples,alpha=0.89)))
print('97% hpd: '+str(pm.hpd(samples,alpha=0.97)))

dummy_w = binom.rvs(n=100,p=samples,size=sample_size)
_,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(posterior)
ax1.hist(dummy_w,bins=50)
plt.show()
means = [(dummy_w == i).mean() for i in range(100)]

print('boys = %f from number of births %f '%(np.sum(birth1),len(birth1)))
plt.plot(means)
plt.show()


