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

plt.plot(d2.height, d2.weight, '.')
plt.title('height to weights')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu = alpha + beta * d2.weight
    # pm.Deterministic('mu', alpha + beta * d2.weight)
    h = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace_model = pm.sample(1000, tune=1000)

pm.traceplot(trace_model)
plt.show()
print(pm.summary(trace_model, alpha=0.11))

trace_df = pm.trace_to_dataframe(trace_model)
print(trace_df.corr().round(2))

# exp
# with pm.Model() as model:
#     alpha = pm.Normal('alpha', mu=178, sd=100)
#     beta = pm.Normal('beta', mu=0, sd=10)
#     sigma = pm.Uniform('sigma', lower=0, upper=50)
#     mu = alpha * np.exp(-beta * d2.weight)
#     # pm.Deterministic('mu', alpha + beta * d2.weight)
#     h = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
#     trace_model = pm.sample(1000, tune=1000)
#
# pm.traceplot(trace_model)
# plt.show()
# print(pm.summary(trace_model, alpha=0.11))

d2 = d2.assign(weight_c=pd.Series(d2.weight - d2.weight.mean()))

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu = alpha + beta * d2.weight_c
    # pm.Deterministic('mu', alpha + beta * d2.weight)
    h = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace_model = pm.sample(1000, tune=1000)

pm.traceplot(trace_model)
plt.show()
print(pm.summary(trace_model, alpha=0.11))

trace_df = pm.trace_to_dataframe(trace_model)
print(trace_df.corr().round(2))

plt.plot(d2.weight_c, d2.height, '.')
plt.plot(d2.weight_c, trace_model['alpha'].mean() + trace_model['beta'].mean() * d2.weight_c)
plt.xlabel(d2.columns[1])
plt.ylabel(d2.columns[0])
plt.show()

post = pm.trace_to_dataframe(trace_model)[:5]
print(post)

N = [10, 50, 150, 352][0]
with pm.Model() as m_N:
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu = alpha + beta * d2.weight_c[:N]
    # pm.Deterministic('mu', alpha + beta * d2.weight)
    h = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height[:N])
    trace_N = pm.sample(1000, tune=1000)

chain_N = trace_N[100:]
pm.traceplot(chain_N)
plt.show()

plt.plot(d2.weight_c[:N], d2.height[:N], 'C0o')
for _ in range(0, 20):
    idx = np.random.randint(len(chain_N))
    plt.plot(d2.weight_c[:N], chain_N['alpha'][idx] + chain_N['beta'][idx] * d2.weight_c[:N], 'C2-', alpha=0.5)
plt.xlabel(d2.columns[1])
plt.ylabel(d2.columns[0])
plt.show()
