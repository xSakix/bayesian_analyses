import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/Howell1.csv', sep=';', header=0)
print(d.head())
print(d.height)
d2 = d[d.age >= 18]
print(d2)

sns.kdeplot(d2.height)
plt.title('heights')
plt.show()

x = np.linspace(100, 250, 100)
plt.plot(x, norm.pdf(x, 178, 20))
plt.title('gausian mean = 178, var=20 (so heights from 178 +- 20 cm)')
plt.show()

x = np.linspace(-10, 60, 100)
plt.plot(x, uniform.pdf(x, 0, 50))
plt.title('variance from 0-50')
plt.show()

sample_mu = norm.rvs(size=10000, loc=178, scale=20)
sample_sigma = uniform.rvs(size=10000, loc=0, scale=50)
prior_h = norm.rvs(loc=sample_mu, scale=sample_sigma)

sns.kdeplot(prior_h)
plt.title('plausibility distribution of heights with top at ' + str(np.mean(prior_h)) + ' cm')
plt.show()

post = np.mgrid[140:160:0.1, 4:9:0.1].reshape(2, -1).T
likehood = [sum(norm.logpdf(d2.height, loc=post[:, 0][i], scale=post[:, 1][i])) for i in range(len(post))]
post_prod = (likehood + norm.logpdf(post[:, 0], loc=178, scale=20) + uniform.logpdf(post[:, 1], loc=0, scale=50))
post_prob = np.exp(post_prod - max(post_prod))

plt.plot(post_prob)
plt.title('grid aproximation of heights')
plt.show()

xi = np.linspace(post[:, 0].min(), post[:, 0].max(), 100)
yi = np.linspace(post[:, 1].min(), post[:, 1].max(), 100)
zi = griddata((post[:, 0], post[:, 1]), post_prob, (xi[None, :], yi[:,None]))
plt.contour(xi, yi, zi)
plt.title('contour of heights')
plt.show()

sample_rows = np.random.choice(np.arange(len(post)),size=10000,replace=True,p=(post_prob/post_prob.sum()))
sample_mu = post[:,0][sample_rows]
sample_sigma = post[:,1][sample_rows]

plt.plot(sample_mu,sample_sigma,'o',alpha=0.05)
plt.axis('equal')
plt.grid(False)
plt.title('heatmap of heights')
plt.show()

_,ax = plt.subplots(1,2)
pm.kdeplot(sample_mu,ax=ax[0])
ax[0].set_title('sample_mu')
pm.kdeplot(sample_sigma,ax=ax[1])
ax[1].set_title('sample_sigma')
plt.show()

print(pm.hpd(sample_mu))
print(pm.hpd(sample_sigma))

d3 = np.random.choice(d2.height,size=20)
post2 = np.mgrid[150:170:0.1, 4:20:0.1].reshape(2, -1).T
likehood2 = [sum(norm.logpdf(d2.height, loc=post2[:, 0][i], scale=post2[:, 1][i])) for i in range(len(post2))]
post_prod2 = (likehood2 + norm.logpdf(post2[:, 0], loc=178, scale=20) + uniform.logpdf(post2[:, 1], loc=0, scale=50))
post_prob2 = np.exp(post_prod2 - max(post_prod2))

sample_rows2 = np.random.choice(np.arange(len(post2)),size=10000,replace=True,p=(post_prob2/post_prob2.sum()))
sample_mu2 = post2[:,0][sample_rows2]
sample_sigma2 = post2[:,1][sample_rows2]

plt.plot(sample_mu2,sample_sigma2,'o',alpha=0.05)
plt.axis('equal')
plt.grid(False)
plt.title('heatmap of heights 2(small sample size)')
plt.show()


_,ax = plt.subplots(1,2)
pm.kdeplot(sample_mu2,ax=ax[0])
ax[0].set_title('sample_mu2')
pm.kdeplot(sample_sigma2,ax=ax[1])
ax[1].set_title('sample_sigma2')
plt.show()



