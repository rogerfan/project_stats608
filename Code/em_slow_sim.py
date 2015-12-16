from time import process_time

import numpy as np
from scipy.stats import multivariate_normal as mnorm

from utilities import calc_probs, calc_pdfs, calc_loglik, sample_multinomials


def em_alg(data, init_mu, init_sigma, init_mix, beta_func=None,
           num_iter=100, verbose=False):
    '''
    Estimate Gaussian mixture models with EM algorithm.

    Parameters
    ----------
    data : 2-d array
        Data to cluster.
    init_mu : 2-d array
        Array of initial means. init_mu[k] should provide the mean vector
        for the kth group.
    init_sigma : list of 2-d arrays or 3-d array
        Initial covariance matrices. init_sigma[k] should provide the
        covariance matrix for the kth group.
    init_mix : array
        Initial mixing components.
    num_iter : int, optional
        Number of EM iterations to run. (default 100)
    verbose : bool, optional
        Set to true to print status updates.

    Returns
    -------
    res : (curr_mu, curr_sigma, curr_mix)
        Final estimates of the mean, covariance, and mixture components.
    diag : (logliks, times)
        Diagnostics by iteration.

    '''
    if len(init_mu) != len(init_sigma) or len(init_sigma) != len(init_mix):
        raise ValueError(
            'Number of initial values needs to be consistent.')
    if not np.isclose(sum(init_mix), 1):
        raise ValueError(
            'Initial mixing components should add to 1.')

    if beta_func is None:
        beta_func = lambda i: 1.

    num_groups = len(init_mu)
    n = len(data)
    try:
        p = len(init_mu[0])
    except TypeError:
        p = 1

    curr_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    curr_mix = init_mix.copy()
    logliks = np.zeros(num_iter+1)
    time_iter = np.zeros(num_iter+1)

    start_time = process_time()
    for iternum in range(num_iter):
        if verbose:  # Status updates
            if np.isclose(iternum // 100, iternum / 100) and iternum != 0:
                print()
            if np.isclose(iternum // 10, iternum / 10) and iternum != 0:
                print('.', end='', flush=True)
        beta = beta_func(iternum)
        if beta > 1.:
            beta = 1.

        # E-step
        start = process_time()
        pdfs = calc_pdfs(data, curr_mu, curr_sigma)
        probs = calc_probs(pdfs, curr_mix, beta)
        time_iter[iternum+1] += process_time() - start

        probs_raw = calc_probs(pdfs, curr_mix, 1.)
        logliks[iternum] = calc_loglik(data, pdfs, probs)
        # logliks[iternum] = calc_loglik(data, pdfs, probs_raw)

        # M-step
        start = process_time()
        for k in range(num_groups):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])
            curr_sigma[k] = np.cov(data, rowvar=0, aweights=probs[:,k], ddof=0)
        curr_mix = np.mean(probs, axis=0)
        time_iter[iternum+1] += process_time() - start

    logliks[-1] = calc_loglik(
        data,
        calc_pdfs(data, curr_mu, curr_sigma),
        calc_probs(pdfs, curr_mix, 1.)
    )
    times = np.cumsum(time_iter)

    return logliks, times

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import colors

    n = 2000
    np.random.seed(29643)

    # overlap contrast
    num_groups = 3
    mu = np.array([[0., 2.], [2., 1.], [2., 3.]])
    sigma = np.array(
        [[[1.0,  0.1], [ 0.1, 0.3]],
         [[1.0, -0.1], [-0.1, 1.0]],
         [[0.5, -0.5], [-0.5, 1.0]]])
    mix = np.array([.15,.7,.15])
    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Estimate
    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3
    niter = 500
    logliks_em, _ = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=niter)
    i=0
    run_storage = run_storage = np.zeros(((niter+1)*2, 3))
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 0] = i
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 1] = np.arange(niter+1)
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 2] = logliks_em


    np.random.seed(29643)
    mu = np.array([[0., 2.], [2., 0.], [3., 4.]])
    print(mix)
    print('mix total: ', np.sum(mix))

    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Estimate
    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3
    niter = 500
    logliks_em, _ = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=niter)
    i=1
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 0] = i
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 1] = np.arange(niter+1)
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 2] = logliks_em

    np.savetxt(
        './intermediate_data/em_mean.csv', run_storage,
        header='mix, iter, loglik_curr',
        delimiter=',', comments='')


    # mixture contrast
    num_groups = 3
    mu = np.array([[0., 2.], [2., 1.], [2., 3.]])
    sigma = np.array(
        [[[1.0,  0.1], [ 0.1, 0.3]],
         [[1.0, -0.1], [-0.1, 1.0]],
         [[0.5, -0.5], [-0.5, 1.0]]])
    mix = np.array([4,4.5,3.75])
    mix = mix / np.sum(mix)
    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Estimate
    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3
    niter = 500
    logliks_em, _ = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=niter)
    i=0
    run_storage = run_storage = np.zeros(((niter+1)*2, 3))
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 0] = i
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 1] = np.arange(niter+1)
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 2] = logliks_em


    np.random.seed(29643)
    mu = np.array([[0., 2.], [2., 0.], [3., 4.]])

    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Estimate
    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3
    niter = 500
    logliks_em, _ = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=niter)
    i=1
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 0] = i
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 1] = np.arange(niter+1)
    run_storage[i*(niter + 1):(i+1)*(niter + 1), 2] = logliks_em

    np.savetxt(
        './intermediate_data/em_mix.csv', run_storage,
        header='mix, iter, loglik_curr',
        delimiter=',', comments='')