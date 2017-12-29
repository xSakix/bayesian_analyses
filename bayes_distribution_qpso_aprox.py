import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import sys

sys.path.insert(0, '../qpso_curve_fit')
from qpso_curve_fit import QPSOCurveFit

import sys

sys.path.insert(0, '../rebalancer')
from rebalancer import load_data

import price_changes

data, data2 = load_data(['TECL'], '2010-01-01', '2017-12-12')

list_of_price_changes = price_changes.compute(data)

c1 = list_of_price_changes.count(1)
c0 = list_of_price_changes.count(0)

print(c1)
print(c0)
sample_size = len(list_of_price_changes)
p1 = c1 / sample_size
p0 = c0 / sample_size
print(p1)
print(p0)

print('likehood(goes up, binom):' + str(binom.pmf(c1, sample_size, p1)))
print(binom.pmf(c0, sample_size, p0))

x = np.linspace(0., 1., sample_size)
priors = np.random.uniform(0., 1., sample_size)
priors.sort()

binom_data = binom.pmf(c1, sample_size, x)
t = binom_data * priors
t = t / np.sum(t)

qpso = QPSOCurveFit(400, 200, m=3)

result = qpso.run(binom_data, t)

posterior = result.evaluate(binom_data)

plt.plot(priors, t)
plt.plot(priors, posterior)
plt.legend(['grid_aprox', 'quadratic'])
plt.show()
