import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from crypto_quandl_loader import load_data
from qa import quadratic_approximation
import price_changes

data = load_data('BTER/VTCBTC')
diffs = np.linspace(0.01, 0.1, 10)

legends = []

for diff in diffs:
    list_of_price_events = price_changes.compute_with_difference(diff, data)

    c1 = list_of_price_events.count(1)
    c0 = list_of_price_events.count(0)

    sample_size = len(list_of_price_events)
    p1 = c1 / sample_size
    p0 = c0 / sample_size

    print('likehood(diff of price >= %f):%f' % (diff, binom.pmf(c1, sample_size, p1)))
    print('likehood(diff of price < %f):%f' % (diff, binom.pmf(c0, sample_size, p0)))

    priors = np.random.uniform(0., 1., sample_size)
    priors.sort()

    mean_q, std_q = quadratic_approximation(sample_size, c1)

    diff_str = str(diff)
    plt.plot(priors, norm.pdf(priors, mean_q['p'], std_q), label=diff_str)

plt.xlabel('chance diff change in price of VTC-BTC pair')
plt.ylabel('density')
plt.title('quadratic approximation')
plt.legend(loc=0, fontsize=13)

plt.show()
