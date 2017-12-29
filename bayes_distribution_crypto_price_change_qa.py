import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom,beta,norm
from crypto_quandl_loader import load_data
from qa import quadratic_approximation
import price_changes


data = load_data('BTER/VTCBTC')
list_of_price_changes = price_changes.compute(data)

c1 = list_of_price_changes.count(1)
c0 = list_of_price_changes.count(0)

sample_size = len(list_of_price_changes)
p1 = c1 / sample_size
p0 = c0 / sample_size

print('likehood(goes up, binom):' + str(binom.pmf(c1, sample_size, p1)))
print('likehood(goes down, binom):%f' % binom.pmf(c0, sample_size, p0))

priors = np.random.uniform(0., 1., sample_size)
priors.sort()

mean_q, std_q = quadratic_approximation(sample_size,c1)


plt.plot(priors, beta.pdf(priors, c1 + 1, sample_size - c1 + 1), label='beta posterior')
plt.plot(priors, norm.pdf(priors, mean_q['p'], std_q), label='quadratic approximation')
plt.legend(loc=0, fontsize=13)
plt.xlabel('chance VTC-BTC goes up in price')
plt.ylabel('density')
plt.show()
