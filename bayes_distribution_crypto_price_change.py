import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from crypto_quandl_loader import load_data
import price_changes
import grid_approx

data = load_data('BTER/VTCBTC')
up_down = price_changes.compute(data)

c1 = up_down.count(1)
c0 = up_down.count(0)

p1 = c1 / len(up_down)
p0 = c0 / len(up_down)

print('likehood(goes up, binom):' + str(binom.pmf(c1, len(up_down), p1)))
print(binom.pmf(c0, len(up_down), p0))

p_grid, posterior = grid_approx.compute(up_down, 1)

plt.plot(p_grid, posterior)
plt.show()
