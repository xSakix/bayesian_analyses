import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from crypto_quandl_loader import load_data
import price_changes

data = load_data('BTER/VTCBTC')

diffs = np.linspace(0.1, 1., 10)

legends = []

for diff in diffs:
    up_down = price_changes.compute_with_difference(diff, data)

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
