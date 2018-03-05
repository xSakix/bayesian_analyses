# multivariate lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/WaffleDivorce.csv', sep=';', header=0)

plt.plot(d.MedianAgeMarriage,d.Divorce,'C0o')
plt.xlabel('median age marriage')
plt.ylabel('divorce')
plt.show()

d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - np.mean(d.MedianAgeMarriage)) / np.std(d.MedianAgeMarriage)

with pm.Model() as model:
    sigma = pm.Uniform(name='sigma', lower=0, upper=10)
    bA = pm.Normal(name='bA', mu=0, sd=1)
    a = pm.Normal(name='a', mu=10, sd=10)
    mu = pm.Deterministic('mu', a + bA * d.MedianAgeMarriage_s)
    divorce = pm.Normal(name='divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_model = pm.sample(1000, tune=1000)

pm.traceplot(trace_model)
plt.show()

mu_mean = trace_model['mu']
mu_hpd = pm.hpd(mu_mean, alpha=0.11)

plt.plot(d.MedianAgeMarriage_s, d.Divorce, 'C0o')
plt.plot(d.MedianAgeMarriage_s, mu_mean.mean(0), 'C2')

idx = np.argsort(d.MedianAgeMarriage_s)
plt.fill_between(d.MedianAgeMarriage_s[idx], mu_hpd[:, 0][idx], mu_hpd[:, 1][idx], color='C2', alpha=0.25)

#plt.fill_between(d.MedianAgeMarriage_s, mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)

plt.xlabel('median age marriage')
plt.ylabel('divorce')
plt.show()


samples = pm.sample_ppc(trace_model,1000,model)

mu_mean = samples['divorce']
mu_hpd = pm.hpd(mu_mean, alpha=0.11)

plt.plot(d.MedianAgeMarriage_s, d.Divorce, 'C0o')

idx = np.argsort(d.MedianAgeMarriage_s)

plt.plot(d.MedianAgeMarriage_s[idx], mu_mean.mean(0)[idx], 'C2')
#plt.plot(d.MedianAgeMarriage_s, mu_mean.mean(0), 'C2')

plt.fill_between(d.MedianAgeMarriage_s[idx], mu_hpd[:, 0][idx], mu_hpd[:, 1][idx], color='C2', alpha=0.25)

#plt.fill_between(d.MedianAgeMarriage_s, mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)

plt.xlabel('median age marriage')
plt.ylabel('divorce')
plt.show()

