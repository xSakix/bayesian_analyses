# quadratic aproximation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import griddata
import pymc3 as pm

d = pd.read_csv('../../rethinking/data/Howell1.csv', sep=';', header=0)
d2 = d[d.age >= 18]

with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    mean_q = pm.find_MAP()
    std_mu = ((1 / pm.find_hessian(mean_q, vars=[mu])) ** 0.5)[0]
    std_sigma = ((1 / pm.find_hessian(mean_q, vars=[sigma])) ** 0.5)[0]
    trace_model = pm.sample(1000, tune=1000)

print('mean mu:' + str(mean_q['mu']))
print('mean sigma:' + str(mean_q['sigma']))
print('std mu:' + str(std_mu))
print('std sigma:' + str(std_sigma))

samples = norm.rvs(loc=mean_q['mu'], scale=mean_q['sigma'], size=10000)
print('89 percentile:' + str(pm.hpd(samples, alpha=0.89)))
print('95 percentile:' + str(pm.hpd(samples, alpha=0.95)))

pm.summary(trace_model, alpha=0.11)
pm.summary(trace_model, alpha=0.05)

trace_df = pm.trace_to_dataframe(trace_model)
print('covariance of variables:' + str(trace_df.cov()))
print('vector of variances for parametes' + str(np.diag(trace_df.cov())))
print('correlation matrix:' + str(trace_df.corr()))

samples = multivariate_normal.rvs(mean=trace_df.mean(), cov=trace_df.cov(), size=10)
print(samples)

print('-----------------------------------')
with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sd=0.1)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    mean_q = pm.find_MAP()
    std_mu = ((1 / pm.find_hessian(mean_q, vars=[mu])) ** 0.5)[0]
    std_sigma = ((1 / pm.find_hessian(mean_q, vars=[sigma])) ** 0.5)[0]
    trace_model = pm.sample(1000, tune=1000)

print('mean mu:' + str(mean_q['mu']))
print('mean sigma:' + str(mean_q['sigma']))
print('std mu:' + str(std_mu))
print('std sigma:' + str(std_sigma))

samples = norm.rvs(loc=mean_q['mu'], scale=mean_q['sigma'], size=10000)
print('89 percentile:' + str(pm.hpd(samples, alpha=0.89)))
print('95 percentile:' + str(pm.hpd(samples, alpha=0.95)))

pm.summary(trace_model, alpha=0.11)
pm.summary(trace_model, alpha=0.05)

trace_df = pm.trace_to_dataframe(trace_model)
print('covariance of variables:' + str(trace_df.cov()))
print('vector of variances for parametes' + str(np.diag(trace_df.cov())))
print('correlation matrix:' + str(trace_df.corr()))

samples = multivariate_normal.rvs(mean=trace_df.mean(), cov=trace_df.cov(), size=10)
print(samples)
for i in range(len(samples)):
    sns.kdeplot(samples[i])
plt.title('multivariate normal samples')
plt.show()

with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sd=20)
    sigma = pm.Lognormal('sigma', mu=2, tau=0.01)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace_model = pm.sample(1000, tune=1000)

pm.traceplot(trace_model)
plt.show()
pm.summary(trace_model,alpha=0.11)