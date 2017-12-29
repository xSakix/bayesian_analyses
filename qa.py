import pymc3 as pm


def quadratic_approximation(sample_size, num_observed_events_in_samples):
    with pm.Model():
        p = pm.Uniform('p', 0., 1.)
        w = pm.Binomial('w', n=sample_size, p=p, observed=num_observed_events_in_samples)
        mean_q = pm.find_MAP()
        std_q = ((1 / pm.find_hessian(mean_q, vars=[p])) ** 0.5)[0]

    print('mean=%f' % mean_q['p'])
    print('std=%f' % std_q)

    return mean_q,std_q

