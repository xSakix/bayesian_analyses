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
    a = pm.Normal('a', mu=10, sd=10)
    bA = pm.Normal('bA', mu=0, sd=1, shape=2)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bA[0] * d.Marriage_s + bA[1] * d.MedianAgeMarriage_s)
    divorce = pm.Normal('divorce', mu=mu, sd=sigma, observed=d.Divorce)
    trace_5_3 = pm.sample(1000, tune=1000)

trace_df = pm.trace_to_dataframe(trace_5_3)
print(trace_df[['a', 'bA__0', 'bA__1']].corr().round(2))

mu_pred = trace_5_3['mu']
mu_hpd = pm.hpd(mu_pred, alpha=0.05)
divorce_pred = pm.sample_ppc(trace_5_3, model=model, samples=1000)['divorce']
divorce_hpd = pm.hpd(divorce_pred)

plt.errorbar(d.Divorce, divorce_pred.mean(0), yerr=np.abs(divorce_pred.mean(0) - mu_hpd.T), fmt='C0o')
plt.plot(d.Divorce, divorce_pred.mean(0), 'C0o')
plt.xlabel('observed divorce')
plt.ylabel('predicted divorce')
min_x, max_x = d.Divorce.min(), d.Divorce.max()
plt.plot([min_x, max_x], [min_x, max_x], 'k--')
plt.show()

plt.figure(figsize=(10, 12))

residuals = d.Divorce - mu_pred.mean(0)
idx = np.argsort(residuals)
y_label = d.Loc[idx]
y_points = np.linspace(0, 1, 50)

plt.errorbar(residuals[idx],
             y_points,
             xerr=np.abs(divorce_pred.mean(0) - mu_hpd.T),
             fmt='C0o',
             lw=3)

plt.errorbar(residuals[idx],
             y_points,
             xerr=np.abs(divorce_pred.mean(0) - divorce_hpd.T),
             fmt='C0o',
             lw=3,
             alpha=0.5)

plt.yticks(y_points, y_label)
plt.vlines(0, 0, 1, 'grey')
plt.show()
