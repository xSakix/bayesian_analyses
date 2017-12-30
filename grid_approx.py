import numpy as np
from scipy.stats import binom


def compute(observations, event='W', priors=None):
    p_grid = np.linspace(0., 1., len(observations))
    if priors is None:
        priors = np.random.uniform(0., 1., len(observations))
        priors.sort()
    posterior = binom.pmf(observations.count(event), len(observations), p_grid)
    posterior = posterior * priors
    posterior = posterior / posterior.sum()
    return p_grid, posterior
