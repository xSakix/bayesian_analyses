# multivariate lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/WaffleDivorce.csv', sep=';', header=0)

d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - np.mean(d.MedianAgeMarriage)) / np.std(d.MedianAgeMarriage)
d['Marriage_s'] = (d.Marriage - np.mean(d.Marriage)) / np.std(d.Marriage)

with pm.Model() as model:
    sigma = pm.Uniform(name='sigma', lower=0, upper=10)
    b = pm.Normal(name='b', mu=0, sd=1)
    a = pm.Normal(name='a', mu=10, sd=10)
    mu = pm.Deterministic('mu', a + b * d.MedianAgeMarriage_s)
    marriage = pm.Normal(name='marriage', mu=mu, sd=sigma, observed=d.Marriage_s)
    trace_model = pm.sample(1000, tune=1000)

pm.traceplot(trace_model)
plt.show()

mu_mean = trace_model['mu'].mean(0)
residual = d.Marriage_s - mu_mean

idx = np.argsort(d.MedianAgeMarriage_s)
d.plot('MedianAgeMarriage_s', 'Marriage_s', kind='scatter', xlim=(-3, 3), ylim=(-3, 3))
plt.plot(d.MedianAgeMarriage_s[idx], mu_mean[idx], 'k')
plt.vlines(d.MedianAgeMarriage_s, mu_mean, mu_mean + residual, color='grey')
plt.show()
