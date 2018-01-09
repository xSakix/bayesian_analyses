import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import mode
import seaborn as sns


def compute_medium(priors):
    p_grid = np.linspace(0., 1., 1000)

    likehood = binom.pmf(8, 15, p_grid)
    post = likehood * priors
    post = post / post.sum()
    plt.plot(p_grid, post)
    plt.show()
    samples = np.random.choice(p_grid, p=post, size=10000, replace=True)
    sns.kdeplot(samples)
    plt.show()
    print('90% hpdi = ' + str(pm.hpd(samples, alpha=0.9)))
    dummy_w = binom.rvs(n=15, p=samples, size=10000)
    _, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(p_grid, post)
    ax1.hist(dummy_w, bins=50)
    plt.show()
    mean_8_15 = (dummy_w == 8).mean()
    print('prob of 8 success in 15 tosses: ' + str(mean_8_15))
    dummy_w = binom.rvs(n=9, p=samples, size=10000)
    _, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(p_grid, post)
    ax1.hist(dummy_w, bins=50)
    plt.show()
    mean_6_9 = (dummy_w == 6).mean()
    print('prob of 6 success in 9 tosses(using samples for 8 in 15): ' + str(mean_6_9))


def make_priors(size):
    priors = np.random.uniform(0., 1., size)
    priors[priors < 0.5] = 0.
    priors[priors >= 0.5] = 1.
    if priors.sum() == 0.:
        return make_priors(size)
    return priors


compute_medium(np.repeat(1., 1000))
compute_medium(np.array(make_priors(1000)))
