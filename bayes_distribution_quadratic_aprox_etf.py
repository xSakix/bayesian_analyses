import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import beta
import sys
from qa import quadratic_approximation

sys.path.insert(0, '../rebalancer')
from rebalancer import load_data

import price_changes

data, data2 = load_data(['TECL'], '2010-01-01', '2017-12-12')

price_change_list = price_changes.compute(data)

c1 = price_change_list.count(1)
c0 = price_change_list.count(0)

print(c1)
print(c0)
sample_size = len(price_change_list)
p1 = c1 / sample_size
p0 = c0 / sample_size
print(p1)
print(p0)

print('likehood(goes up, binom):' + str(binom.pmf(c1, sample_size, p1)))
print('likehood(goes down, binom):%f' % binom.pmf(c0, sample_size, p0))

priors = np.random.uniform(0., 1., sample_size)
priors.sort()

mean_q, std_q = quadratic_approximation(sample_size,c1)


plt.plot(priors, beta.pdf(priors, c1 + 1, sample_size - c1 + 1), label='beta posterior')
plt.plot(priors, norm.pdf(priors, mean_q['p'], std_q), label='quadratic approximation')
plt.legend(loc=0, fontsize=13)
plt.xlabel('chance TECL goes up in price')
plt.ylabel('density')
plt.show()
