import numpy as np
import etf_data_loader
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import binom


def yorke(x, r):
    return x * r * (1. - x)


class Individual:

    def __init__(self, num_of_assets=1, r=np.random.uniform(2.9, 4.0, 2)):
        self.cash = 0.
        self.invested = 0.
        self.history = []
        self.invested_history = []
        self.shares = []
        self.r = r
        if num_of_assets == 1:
            self.dist = np.array([0.5])
        else:
            self.dist = np.full(num_of_assets, 0.9 / num_of_assets)
        self.events = []
        self.num_of_assets = num_of_assets
        self.tr_cost = 2.
        self.x_history = []

    def load_high_low(self, df_high, df_low, df_open):
        if df_high is not None and df_low is not None:
            df_high_low = np.abs(df_high - df_low)
        else:
            for key in df_open.keys():
                std = df_open[key].std()
                mean = df_open[key].mean()

        return df_high_low

    def simulate(self, df_open, df_high, df_low):
        x = np.array([0.01, 0.01])
        for _ in range(20):
            x = yorke(x, self.r)
            self.x_history.append(x)

        df_high_low = self.load_high_low(df_high, df_low, df_open)

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
            if x[0] >= 0.9:
                self.cash = np.dot(self.shares, prices) - self.num_of_assets * self.tr_cost
                self.events.append('S')
            # buy
            if x[1] >= 0.9:
                c = np.multiply(self.dist, portfolio)
                c = np.subtract(c, self.tr_cost)
                s = np.divide(c, prices)
                s = np.floor(s)
                self.cash = portfolio - np.dot(self.shares, prices) - self.num_of_assets * self.tr_cost
                self.shares = s
                self.events.append('B')

            if x[0] < 0.9 and x[1] < 0.9:
                self.events.append('H')

            day += 1

    def compute_prices(self, high_low, prices):
        for i in range(len(high_low)):
            if high_low[i] <= 0. or np.isnan(high_low[i]):
                high_low[i] = 0.001
        noise = np.random.normal(0, high_low)
        # print('noise = '+str(noise))
        # print('price before = '+str(prices))
        prices = np.add(prices, noise)
        # print('price after = '+str(prices))

        return prices


etf = ['BND', 'SPY', 'XLK', 'VWO']
start_date = '2010-01-01'
end_date = '2017-12-12'

df_open, df_close, df_high, df_low, df_adj_close = etf_data_loader.load_data(etf, start_date, end_date)

best = None
for _ in range(50):
    ind = Individual(len(etf))
    ind.simulate(df_adj_close, df_high, df_low)
    if best is None:
        best = ind

    if best.history[-1] < ind.history[-1]:
        print(best.history[-1])
        best = ind

returns = (best.history[-1] - best.invested_history[-1]) / best.invested_history[-1]
print('value: ' + str(best.history[-1]))
print('invested:' + str(best.invested_history[-1]))
print('returns: ' + str(returns))

_, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(best.history)
ax1.hist(best.events, bins=50)
plt.show()

size=len(best.events)
priors = np.random.uniform(0.,1.,size)
p_grid = np.linspace(0.,1.,size)
likehood = binom.pmf(best.events.count('B'),size, p=p_grid)
posterior = likehood*priors
posterior = posterior/posterior.sum()
plt.plot(p_grid,posterior)
plt.show()

