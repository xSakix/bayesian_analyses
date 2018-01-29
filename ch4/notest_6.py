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

plt.plot(d2.height,d2.weight,'.')
plt.title('height to weights')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha',mu=178,sd=100)
    beta = pm.Normal('beta',mu=0,sd=10)
    sigma = pm.Uniform('sigma',lower=0,upper=50)
    mu = alpha+beta*d2.weight
    h = pm.Normal('height',mu=mu,sd=sigma,observed=d2.height)
    trace_model = pm.sample(1000,tune=1000)

pm.traceplot(trace_model)
plt.show()
pm.summary(trace_model,alpha=0.11)
