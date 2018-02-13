# lin regresion of heights vs weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/Howell1.csv', sep=';', header=0)
d2 = d[d.age >= 18]

d2 = d2.assign(weight_c=pd.Series(d2.weight - d2.weight.mean()))

N = 10

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu = pm.Deterministic('mu', alpha + beta * d2.weight[:N])
    h = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height[:N])
    trace_model = pm.sample(1000, tune=1000)

trace_model = trace_model[100:]

mu_at_50 = trace_model['alpha'] + trace_model['beta'] * 50.
pm.kdeplot(mu_at_50)
plt.xlabel('heights')
plt.yticks([])
plt.show()

# 89% of ways the model produces the data place the average height at 159-160cm when the wieght
# is 50kg
print(pm.hpd(mu_at_50, alpha=0.11))

df_trace_N = pm.trace_to_dataframe(trace_model).filter(regex=('mu.*'))
print(df_trace_N.head())

weight_seq = np.arange(25., 71.)
trace_model_thinned = trace_model[::10]
mu_pred = np.zeros((len(weight_seq), len(trace_model_thinned) * trace_model.nchains))
for i, w in enumerate(weight_seq):
    mu_pred[i] = trace_model_thinned['alpha'] + trace_model_thinned['beta'] * w

plt.plot(weight_seq, mu_pred, 'C0.', alpha=0.1)
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

mu_mean = mu_pred.mean(1)
mu_hpd = pm.hpd(mu_pred.T, alpha=0.11)
print('mean mu: ' + str(mu_mean))
print('mu hpd 89%: ' + str(mu_hpd))

_, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(weight_seq, mu_pred, 'C0.', alpha=0.1)
ax1.scatter(d2.weight[:N], d2.height[:N])
ax1.plot(weight_seq, mu_mean, 'C2')
ax1.fill_between(weight_seq, mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)
plt.show()
