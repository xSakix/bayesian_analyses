# multivariate lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/WaffleDivorce.csv', sep=';', header=0)

plt.plot(d.MedianAgeMarriage, d.Divorce, 'C0o')
plt.xlabel('median age marriage')
plt.ylabel('divorce')
plt.show()

plt.plot(d.Marriage, d.Divorce, 'C0o')
plt.xlabel('marriage')
plt.ylabel('divorce')
plt.show()

d['Marriage_s'] = (d.Marriage - np.mean(d.Marriage)) / np.std(d.Marriage)
d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - np.mean(d.MedianAgeMarriage)) / np.std(d.MedianAgeMarriage)

with pm.Model() as model:
    sigma = pm.Uniform(name='sigma', lower=0, upper=10)
    bA = pm.Normal(name='bA', mu=0, sd=1)
    bR = pm.Normal(name='bR', mu=0, sd=1)
    a = pm.Normal(name='a', mu=10, sd=10)
    mu = pm.Deterministic('mu', a + bR * d.Marriage_s + bA * d.MedianAgeMarriage_s)
    divorce = pm.Normal(name='divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_model = pm.sample(1000, tune=1000)

varnames = ['a', 'bA', 'bR', 'sigma']
pm.traceplot(trace_model, varnames)
plt.show()

print(pm.summary(trace_model,varnames,alpha=0.11))

pm.forestplot(trace_model)
plt.show()