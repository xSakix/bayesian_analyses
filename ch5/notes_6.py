# multivariate lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/WaffleDivorce.csv', sep=';', header=0)

d['Marriage_s'] = (d.Marriage - np.mean(d.Marriage)) / np.std(d.Marriage)
d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - np.mean(d.MedianAgeMarriage)) / np.std(d.MedianAgeMarriage)

with pm.Model() as model:
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    bA = pm.Normal('bA', mu=0, sd=1)
    bR = pm.Normal('bR', mu=0, sd=1)
    a = pm.Normal('a', mu=10, sd=10)
    mu = pm.Deterministic('mu', a + bR * d.Marriage_s + bA * d.MedianAgeMarriage_s)
    divorce = pm.Normal('divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_model = pm.sample(1000, tune=1000)

mu_pred = trace_model['mu']
mu_hpd = pm.hpd(mu_pred, alpha=0.05)
divorce_pred = pm.sample_ppc(trace_model, model=model, samples=1000)['divorce']
divorce_hpd = pm.hpd(divorce_pred)

plt.errorbar(d.Divorce, divorce_pred.mean(0), yerr=np.abs(divorce_pred.mean(0) - mu_hpd.T), fmt='C0o')
plt.plot(d.Divorce,divorce_pred.mean(0),'C0o')
plt.show()