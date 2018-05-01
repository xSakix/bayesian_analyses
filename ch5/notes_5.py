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

R_avg = np.linspace(-3, 3, 100)
mu_pred = trace_model['a'] + trace_model['bR'] * R_avg[:, None]
mu_hpd = pm.hpd(mu_pred.T)
divorce_hpd = pm.hpd(norm.rvs(mu_pred, trace_model['sigma']).T)

mu_pred2 = trace_model['a'] + trace_model['bA'] * R_avg[:, None]
mu_hpd2 = pm.hpd(mu_pred2.T)
divorce_hpd2 = pm.hpd(norm.rvs(mu_pred2, trace_model['sigma']).T)

_, (ax0, ax1) = plt.subplots(1, 2)

ax0.plot(R_avg, mu_pred.mean(1), 'C0')
ax0.fill_between(R_avg, mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)
ax0.fill_between(R_avg, divorce_hpd[:, 0], divorce_hpd[:, 1], color='C2', alpha=0.25)
ax0.set_xlabel('Marriage.s')
ax0.set_ylabel('Divorce')
ax0.set_title('MedianAgeMarriage_s = 0')

ax1.plot(R_avg, mu_pred2.mean(1), 'C0')
ax1.fill_between(R_avg, mu_hpd2[:, 0], mu_hpd2[:, 1], color='C2', alpha=0.25)
ax1.fill_between(R_avg, divorce_hpd2[:, 0], divorce_hpd2[:, 1], color='C2', alpha=0.25)
ax1.set_xlabel('MedianAgeMarriage.s')
ax1.set_ylabel('Divorce')
ax1.set_title('Marriage_s = 0')

plt.show()
