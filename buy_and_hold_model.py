import sys
import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file
import bah_simulator as bah
import random
from datetime import datetime


def gen_random_date(year_low, year_high):
    y = random.randint(year_low, year_high)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


def run_bah_sim(etf):
    start_date = gen_random_date(1993, 2017)
    end_date = gen_random_date(1993, 2017)
    if start_date > end_date:
        tmp = start_date
        start_date = end_date
        end_date = tmp

    #print('%s - %s'%(str(start_date),str(end_date)))

    df_adj_close = load_all_data_from_file('etf_data_adj_close.csv', str(start_date), str(end_date))
    dca = bah.DCA(30, 300.)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(df_adj_close[etf])
    investor.compute_means()

    if investor.m == 0.:
        investor.m = investor.ror_history[-1]

    print('%f:%f:%f' % (investor.invested, investor.history[-1], investor.m))

    return investor


investors = []
while len(investors) < 100:
    try:
        investors.append(run_bah_sim(['FBT']))
    except:
        continue

means = [investor.m for investor in investors]
devs = [investor.std for investor in investors]
plt.plot(means)
plt.show()

with pm.Model() as model:
    mu = pm.Normal('mu', mu=np.mean(means), sd=np.std(means))
    sigma = pm.Uniform('sigma', lower=np.min(devs), upper=np.max(devs))
    mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(means))
    trace_model = pm.sample(1000, tune=1000)

summary = pm.summary(trace_model)
print(summary)

samples = pm.sample_ppc(trace_model, size=100, model=model)

hpd89 = pm.hpd(samples['mean_returns'], alpha=0.89)
hpd95 = pm.hpd(samples['mean_returns'], alpha=0.95)

print('mean 89 percentile:' + str(np.mean(hpd89)))
print('mean 95 percentile:' + str(np.mean(hpd95)))

plt.plot(means)
plt.title('Means of returns gained from Chaos simulation')
plt.show()

sns.kdeplot(samples['mean_returns'][:, 0])
sns.kdeplot(samples['mean_returns'][:, 1])
plt.title('Distribution of mean returns from Chaos simulation')
plt.show()

plt.plot(hpd89[:, 0])
plt.plot(hpd89[:, 1])
plt.title('sampled means bounded by 89 percentile')
plt.legend(['mean', 'lower 89%', 'upper 89%'])
plt.show()

# print(np.mean(data))
# print(np.std(data))
#
# d1 = data[:int(len(data)/2)]
# d_obs = data[int(len(data)/2)+1:]
# d2 = np.power(d1, 2)
# d3 = np.power(d1, 3)
#
# with pm.Model() as model:
#     alpha = pm.Normal(name='alpha', mu=np.mean(data), sd=np.std(data))
#     sigma = pm.Uniform(name='sigma', lower=0, upper=np.std(data))
#     beta = pm.Normal(name='beta', mu=0, sd=10,shape=2)
#     #mu = pm.Deterministic('mu',alpha + beta[0] * d1 + beta[1] * d2 + beta[2] * d3)
#     mu = pm.Deterministic('mu',alpha + beta[0] * d1 + beta[1] * d2)
#     # mu = pm.Deterministic('mu', alpha + beta * d1)
#     price = pm.Normal(name='price', mu=mu, sd=sigma, observed=d_obs)
#     trace = pm.sample(1000, tune=2000)
#
# samples = pm.sample_ppc(trace, 200, model)
# data_new = np.mean(pm.hpd(samples['price'],alpha=0.11))
# print(data_new)
# plt.plot(pm.hpd(samples['price'],alpha=0.11)[:,0])
# plt.plot(pm.hpd(samples['price'],alpha=0.11)[:,1])
# plt.plot(data_new)
# plt.show()
