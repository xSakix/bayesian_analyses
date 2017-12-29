import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import sys

sys.path.insert(0, '../rebalancer')
from rebalancer import load_data

import price_changes
import grid_approx

data, data2 = load_data(['TECL'], '2010-01-01', '2017-12-12')

list_of_price_changes = price_changes.compute(data)

c1 = list_of_price_changes.count(1)
c0 = list_of_price_changes.count(0)

print(c1)
print(c0)
p1 = c1 / len(list_of_price_changes)
p0 = c0 / len(list_of_price_changes)
print(p1)
print(p0)

print('likehood(goes up, binom):' + str(binom.pmf(c1, len(list_of_price_changes), p1)))
print(binom.pmf(c0, len(list_of_price_changes), p0))

p_grid,posterior = grid_approx.compute(list_of_price_changes,1)

plt.plot(p_grid, posterior, 'b')
plt.show()
