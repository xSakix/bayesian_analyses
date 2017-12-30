import grid_approx
import matplotlib.pyplot as plt
import numpy as np


def make_priors(size):
    priors = np.random.uniform(0., 1., size)
    priors[priors < 0.5] = 0.
    priors[priors >= 0.5] = 1.
    if priors.sum() == 0.:
        return make_priors(size)
    return priors.sort()


print(make_priors(10))
# for W
# 1
p_grid, posterior = grid_approx.compute(['W', 'W', 'W'], priors=make_priors(3))
plt.plot(p_grid, posterior, label='#1')

# 2
p_grid, posterior = grid_approx.compute(['W', 'W', 'W', 'L'], priors=make_priors(4))
plt.plot(p_grid, posterior, label='#2')
# 3
p_grid, posterior = grid_approx.compute(['L', 'W', 'W', 'L', 'W', 'W', 'W'], priors=make_priors(7))
plt.plot(p_grid, posterior, label='#3')

plt.xlabel('probability of W')
plt.ylabel('plausibility')
plt.legend()
plt.show()
plt.clf()

# for L
# 1
p_grid, posterior = grid_approx.compute(['W', 'W', 'W'], 'L', priors=make_priors(3))
plt.plot(p_grid, posterior, label='#1')
# 2
p_grid, posterior = grid_approx.compute(['W', 'W', 'W', 'L'], 'L', priors=make_priors(4))
plt.plot(p_grid, posterior, label='#2')
# 3
p_grid, posterior = grid_approx.compute(['L', 'W', 'W', 'L', 'W', 'W', 'W'], 'L', priors=make_priors(7))
plt.plot(p_grid, posterior, label='#3')

plt.xlabel('probability of L')
plt.ylabel('plausibility')
plt.legend()
plt.show()

# 4
for j in range(4):
    obs = ['L', 'W', 'W', 'L', 'W', 'W', 'W']
    for i in range(j * 10):
        obs.append(obs)

    p_grid, posterior = grid_approx.compute(obs, priors=make_priors(len(obs)))
    plt.plot(p_grid, posterior, label='#' + str(len(obs)))

plt.xlabel('probability of W')
plt.ylabel('plausibility')
plt.legend()
plt.show()

# 4L
for j in range(4):
    obs = ['L', 'W', 'W', 'L', 'W', 'W', 'W']
    for i in range(j * 10):
        obs.append(obs)

    p_grid, posterior = grid_approx.compute(obs, 'L', priors=make_priors(len(obs)))
    plt.plot(p_grid, posterior, label='#' + str(len(obs)))

plt.xlabel('probability of L')
plt.ylabel('plausibility')
plt.legend()
plt.show()
