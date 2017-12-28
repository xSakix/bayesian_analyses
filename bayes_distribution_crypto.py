import pandas
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import binom
import quandl


def load_data(assets):
    file = "data_open.csv"
    file2 = "data_close.csv"
    if not os.path.isfile('all_data.csv'):
        panel_data = quandl.get(assets)
        panel_data.to_csv('all_data.csv')
        panel_data.High.to_csv(file)
        panel_data.Low.to_csv(file2)
    data = pandas.read_csv(file)
    data2 = pandas.read_csv(file2)

    del data['Date']
    del data2['Date']

    indexes = []
    for key in data.keys():
        for i in data[key].index:
            val = data[key][i]
            try:
                if np.isnan(val) and not indexes.__contains__(i):
                    indexes.append(i)
            except TypeError:
                if not indexes.__contains__(i):
                    indexes.append(i)
    data.drop(indexes, inplace=True)
    data2.drop(indexes, inplace=True)
    return data, data2


def compute_list_of_changes(d, data):
    up_down = []
    for i in data.index:
        if i == 0:
            continue
        if np.abs(float(data.iloc[i]) - float(data.iloc[i - 1])) / float(data.iloc[i - 1]) >= d:
            up_down.append(1)
        else:
            up_down.append(0)

    return up_down


data, data2 = load_data('BTER/VTCBTC')

diffs = np.linspace(0.1, 1., 10)

legends = []

for diff in diffs:
    up_down = compute_list_of_changes(diff, data)

    c1 = up_down.count(1)
    c0 = up_down.count(0)

    p1 = c1 / len(up_down)
    p0 = c0 / len(up_down)

    print('likehood(goes up, binom):' + str(binom.pmf(c1, len(up_down), p1)))
    print(binom.pmf(c0, len(up_down), p0))

    p_grid = np.linspace(0., 1., len(up_down))
    prior = np.repeat(1., len(up_down))
    likehood = binom.pmf(c1, len(up_down), p_grid)
    posterior = np.multiply(likehood, prior)
    posterior = posterior / np.sum(posterior)

    plt.plot(p_grid, posterior)

    legend = str(diff)
    print(legend)
    legends.append(legend)

plt.legend(legends)

plt.show()
