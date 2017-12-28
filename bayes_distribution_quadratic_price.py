from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import binom


def load_data(assets, start_date, end_date):
    data_source = 'yahoo'
    file = "data_open.csv"
    file2 = "data_close.csv"
    has_to_load_data = False
    if os.path.isfile(file):
        data = pandas.read_csv(file)
        for asset in assets:
            if not data.keys().contains(asset):
                has_to_load_data = True
    if not os.path.isfile(file) or has_to_load_data:
        panel_data = data_reader.DataReader(assets, data_source, start_date, end_date)
        panel_data.to_frame().to_csv('all_data.csv')
        panel_data.ix['Open'].to_csv(file)
        panel_data.ix['Close'].to_csv(file2)
    data = pandas.read_csv(file)
    data2 = pandas.read_csv(file2)
    if data['Date'][0] > data['Date'][len(data['Date']) - 1]:
        rows = []
        rows2 = []
        for i in reversed(data.index):
            row = [data[key][i] for key in data.keys()]
            row2 = [data2[key][i] for key in data2.keys()]
            rows.append(row)
            rows2.append(row2)

        data = pandas.DataFrame(rows, columns=data.keys())
        data2 = pandas.DataFrame(rows2, columns=data2.keys())
    print('Simulation from %s to %s' % (start_date, end_date))
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


data, data2 = load_data(['TECL'], '2010-01-01', '2017-12-12')

up_down = []
for i in data.index:
    if i == 0:
        continue
    if float(data.iloc[i]) > float(data.iloc[i - 1]):
        up_down.append(1)
    else:
        up_down.append(0)

c1 = up_down.count(1)
c0 = up_down.count(0)

print(c1)
print(c0)
p1 = c1 / len(up_down)
p0 = c0 / len(up_down)
print(p1)
print(p0)

print('likehood(goes up, binom):' + str(binom.pmf(c1, len(up_down), p1)))
print(binom.pmf(c0, len(up_down), p0))

p_grid = np.random.uniform(0., 1., len(up_down))
p_grid.sort()

w = binom.pmf(c1, len(up_down), p_grid)
print(w)

# prior = np.repeat(1., len(up_down))
# likehood = binom.pmf(c1, len(up_down), p_grid)
# posterior = np.multiply(likehood, prior)
# posterior = posterior / np.sum(posterior)
#
# plt.plot(p_grid, posterior, 'b')
# plt.show()
