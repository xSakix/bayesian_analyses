# lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/Howell1.csv', sep=';', header=0)

plt.plot(d.weight, d.height, 'C0o')
plt.show()

d.weight_s = (d.weight - d.weight.mean()) / d.weight.std()
d.weight_s2 = d.weight_s ** 2

plt.plot(d.weight_s, d.height, 'C0o')
plt.show()

d.weight_s3 = d.weight_s ** 3

#weight_m = np.vstack((d.weight_s, d.weight_s2, d.weight_s3))

with pm.Model() as model_2:
    sigma = pm.Uniform(name='sigma', lower=0, upper=50)
    beta = pm.Normal(name='beta', mu=0, sd=10, shape=3)
    alpha = pm.Normal(name='alpha', mu=178, sd=100)
    mu = pm.Deterministic('mu', alpha + beta[0]*d.weight_s+beta[1]*d.weight_s2+beta[2]*d.weight_s3)
    h = pm.Normal(name='h', mu=mu, sd=sigma, observed=d.height)
    trace_2 = pm.sample(1000, tune=1000)

varnames = ['alpha', 'beta', 'sigma']
print(pm.summary(trace_2, varnames))
pm.traceplot(trace_2)
plt.show()

mu_pred = trace_2['mu']
idx = np.argsort(d.weight_s)
mu_hpd = pm.hpd(mu_pred, alpha=0.11)[idx]
height_pred = pm.sample_ppc(trace_2, 200, model_2)
height_pred_hpd = pm.hpd(height_pred['h'], alpha=0.11)[idx]

plt.scatter(d.weight_s, d.height, c='C0', alpha=0.3)
plt.fill_between(d.weight_s[idx], mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)
plt.fill_between(d.weight_s[idx], height_pred_hpd[:, 0], height_pred_hpd[:, 1], color='C2', alpha=0.25)
at = np.arange(-2,3)
plt.xticks(at,np.round(at*d.weight.std()+d.weight.mean(),1))
plt.show()
