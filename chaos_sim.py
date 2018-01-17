import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import binom
import sys
import seaborn as sns

sys.path.insert(0, '../rebalancer')
import etf_data_loader


def yorke(x, r):
    return x * r * (1. - x)


class Individual:
    def __init__(self, num_of_assets=1):
        self.cash = 0.
        self.invested = 0.
        self.history = []
        self.invested_history = []
        self.shares = []
        self.r = np.random.uniform(2.9, 4.0, 2)
        if num_of_assets == 1:
            self.dist = np.array([0.5])
        else:
            self.dist = np.full(num_of_assets, 0.9 / num_of_assets)

        self.events = []
        self.num_of_assets = num_of_assets
        self.tr_cost = 2.
        self.x_history = []

    def load_high_low(self, df_high, df_low):
        if df_high is not None and df_low is not None:
            df_high_low = np.abs(df_high - df_low)

        return df_high_low

    def simulate(self, df_open, df_high, df_low):
        x = np.array([0.01, 0.01])
        for _ in range(20):
            x = yorke(x, self.r)
            self.x_history.append(x)

        df_high_low = self.load_high_low(df_high, df_low)

        if len(df_open.keys()) == 0:
            return

        self.shares = np.zeros(len(df_open.keys()), dtype='float64')

        day = 0

        for i in df_open.index:

            if day % 30 == 0:
                self.cash += 300.
                self.invested += 300.

            prices = []
            high_low = []
            for key in df_open.keys():
                prices.append(df_open[key][i])
                high_low.append(df_high_low[key][i])

            prices = np.array(prices, dtype='float64')
            portfolio = self.cash + np.dot(prices, self.shares)

            self.history.append(portfolio)
            self.invested_history.append(self.invested)

            x = yorke(x, self.r)
            self.x_history.append(x)

            prices = self.compute_prices(high_low, prices)

            # sell
            if x[0] >= 0.9 and sum(self.shares > 0) > 0:
                self.cash = np.dot(self.shares, prices) - sum(self.shares > 0) * self.tr_cost
                self.events.append('S')
                self.shares = np.zeros(self.num_of_assets)
            # buy
            if x[1] >= 0.9:
                c = np.multiply(self.dist, portfolio)
                c = np.subtract(c, self.tr_cost)
                s = np.divide(c, prices)
                s = np.floor(s)
                self.shares = s
                self.cash = portfolio - np.dot(self.shares, prices) - self.num_of_assets * self.tr_cost
                self.events.append('B')

            if x[0] < 0.9 and x[1] < 0.9:
                self.events.append('H')

            day += 1

    def compute_prices(self, high_low, prices):
        high_low = np.array(high_low)
        high_low[high_low <= 0.] = 0.001
        high_low[np.isnan(high_low)] = 0.001
        noise = np.random.normal(0, high_low)
        prices = np.add(prices, noise)

        return prices


etf = ['BND', 'SPY', 'XLK', 'VWO']
start_date = '2010-01-01'
end_date = '2017-12-12'

print('Loading data...')
df_open, df_close, df_high, df_low, df_adj_close = etf_data_loader.load_data(etf, start_date, end_date)
print('Data loaded...')

best = None
print('Generating good sim params...')
for i in range(100):
    ind = Individual(len(etf))
    ind.simulate(df_adj_close, df_high, df_low)
    if best is None:
        best = ind

    if best.history[-1] < ind.history[-1]:
        print(ind.history[-1])
        best = ind
print('Simulation data generated...')
print('Evaluating distributions/data...')

returns = (best.history[-1] - best.invested_history[-1]) / best.invested_history[-1]
print('value: ' + str(best.history[-1]))
print('invested:' + str(best.invested_history[-1]))
print('returns: ' + str(returns*100.)+" %")

f,(ax0,ax1) = plt.subplots(1,2)
f.suptitle('development of value')
ax0.plot(best.history)
ax0.plot(best.invested_history)
ax1.plot(best.x_history[:20])
plt.show()


def show_distribution_for(event):

    size = len(best.events)
    #priors = np.random.uniform(0., 1., size)
    #lets asume that any event has an equal prob of occurence before observations are made
    priors = np.repeat(1.,size)
    p_grid = np.linspace(0., 1., size)
    likehood = binom.pmf(best.events.count(event), size, p=p_grid)
    posterior = likehood * priors
    posterior = posterior / posterior.sum()
    plt.figure('posterior distribution of '+event+' events')
    plt.plot(p_grid, posterior)
    plt.show()

    sample_size = int(10000)
    samples = np.random.choice(a=p_grid, p=posterior, size=sample_size)
    plt.figure('samples of '+event+' event distribution')
    sns.kdeplot(samples)
    plt.show()

    print('0.95% = '+str(pm.hpd(samples,alpha=0.95)))


    dummy_w = binom.rvs(n=size, p=samples, size=sample_size)
    f, (ax0, ax1) = plt.subplots(1, 2)
    f.suptitle('posterior from samples of '+event+' distribution and histogram of '+event+' events')
    ax0.plot(posterior)
    ax1.hist(dummy_w, bins=50)
    plt.show()
    means = [(dummy_w == i).mean() for i in range(size)]
    plt.figure('means of sampled model for '+event+' distri')
    plt.plot(means)
    plt.show()

    return dummy_w, samples


# show_distribution_for('B')
# show_distribution_for('S')
data,samples = show_distribution_for('H')
boundaries = pm.hpd(samples,alpha=0.95)
events = []
for s in samples:
    if s >= boundaries[0] or s <= boundaries[1]:
        events.append('H')
    else:
        events.append('B')

print('num of hold events orig:'+str(best.events.count('H')))
print('num of hold events sampled:'+str(events.count('H')))
print('num of buy events orig:'+str(best.events.count('B')))
print('num of buy events sampled:'+str(events.count('B')))

#because there is a lot of Holds, so we could simulate a sample simulation of H. And when not H
# we  do B.

