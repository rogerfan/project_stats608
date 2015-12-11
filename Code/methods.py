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
        logliks[iternum] = calc_loglik(data, pdfs, probs_raw)

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

    return (curr_mu, curr_sigma, curr_mix), (logliks, times)


def sim_anneal(data, init_mu, init_sigma, init_mix, temp_func,
           num_iter=100, seed=None, verbose=False):
    '''
    Estimate Gaussian mixture models with simulated annealing.

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
    temp_func : function
        Temperature function. Should take an iteration number and return
        a temperature.
    num_iter : int, optional
        Number of EM iterations to run. (default 100)
    seed : int, optional
        Random seed to initialize algorithm with.
    verbose : bool, optional
        Set to true to print status updates.

    Returns
    -------
    res : (best_mu, best_sigma, best_mix, best_classes)
        Best estimates of the mean, covariance, and mixture components, as
        well as the best estimated class memberships.
    diag : (logliks, times)
        Diagnostics by iteration.

    '''
    if len(init_mu) != len(init_sigma) or len(init_sigma) != len(init_mix):
        raise ValueError(
            'Number of initial values needs to be consistent.')
    if not np.isclose(sum(init_mix), 1):
        raise ValueError(
            'Initial mixing components should add to 1.')

    num_groups = len(init_mu)
    n = len(data)
    try:
        p = len(init_mu[0])
    except TypeError:
        p = 1
    if seed is not None:
        np.random.seed(seed)

    # Initialize arrays
    curr_mu = init_mu.copy()
    cand_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    cand_sigma = init_sigma.copy()
    curr_mix = init_mix.copy()
    cand_classes = np.zeros((n, num_groups), dtype=int)

    # Randomly assign initial classes
    curr_classes = np.random.multinomial(1, curr_mix, size=n)
    curr_loglik = calc_loglik(
        data, calc_pdfs(data, curr_mu, curr_sigma), curr_classes)
    best_loglik = curr_loglik

    # Initialize storage arrays
    logliks = np.zeros(num_iter+1)
    logliks[0] = curr_loglik
    time_iter = np.zeros(num_iter+1)

    for iternum in range(num_iter):
        if verbose:  # Status updates
            if np.isclose(iternum // 100, iternum / 100) and iternum != 0:
                print()
            if np.isclose(iternum // 10, iternum / 10) and iternum != 0:
                print('.', end='', flush=True)

        start = process_time()

        # Randomly select new candidate classes and calculate the loglik
        pdfs = calc_pdfs(data, curr_mu, curr_sigma)
        probs = calc_probs(pdfs, curr_mix, 1.)

        cand_classes = sample_multinomials(probs)
        for k in range(num_groups):
            cand_mu[k] = np.mean(data[cand_classes[:,k] == 1], axis=0)
            cand_sigma[k] = np.cov(data[cand_classes[:,k] == 1], rowvar=0)
        cand_loglik = calc_loglik(
            data, calc_pdfs(data, cand_mu, cand_sigma), cand_classes)

        # Switching
        accept = np.random.uniform()
        log_accept_prob = (cand_loglik - curr_loglik)/temp_func(iternum)
        if cand_loglik >= curr_loglik or np.log(accept) < log_accept_prob:
            curr_classes = cand_classes
            curr_loglik = cand_loglik
            curr_mu = cand_mu
            curr_sigma = cand_sigma

            curr_mix = np.mean(curr_classes, axis=0)

        # Keep best loglik so far
        if curr_loglik > best_loglik:
            best = (curr_mu, curr_sigma, curr_mix, curr_classes)
            best_loglik = curr_loglik

        # Storage
        time_iter[iternum+1] += process_time() - start
        logliks[iternum+1] = best_loglik

    times = np.cumsum(time_iter)
    return best, (logliks, times)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import colors

    n = 2000
    np.random.seed(29643)

    # Simulate 3 groups
    num_groups = 3
    mu = np.array([[0., 2.], [2., 1.], [2., 3.]])
    sigma = np.array(
        [[[1.0,  0.1], [ 0.1, 0.3]],
         [[1.0, -0.1], [-0.1, 1.0]],
         [[0.5, -0.5], [-0.5, 1.0]]])
    mix = np.array([.15, .7, .15])

    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Plot data
    cmap, norm = colors.from_levels_and_colors(
        levels=[0, 1, 2], colors=['magenta', 'cyan', 'green'], extend='max')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x[:,0], x[:,1], c=z_ind, cmap=cmap, norm=norm)
    fig.savefig('./sim_data.pdf')

    # Estimate
    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3

    res_em, (logliks_em, times_em) = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=250)

    res_da, (logliks_da, times_da) = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=250,
        beta_func=lambda i: 1.-np.exp(-(i+1)/5))

    res_sa, (logliks_sa, times_sa) = sim_anneal(
        x, init_mu, init_sigma, init_mix, num_iter=250, seed=105725,
        temp_func=lambda i: max(1e-4, 100*.992**i))

    # Plotting
    data_mu = np.array(
        [np.mean(x[z_ind == k], axis=0) for k in range(num_groups)])
    data_sigma = np.array(
        [np.cov(x[z_ind == k], rowvar=0) for k in range(num_groups)])

    data_loglik = calc_loglik(
        x, calc_pdfs(x, data_mu, data_sigma), z)
    true_loglik = calc_loglik(x, calc_pdfs(x, mu, sigma), z)

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(logliks_em)
    ax.plot(logliks_da)
    ax.plot(logliks_sa)
    ax.axhline(y=data_loglik, color='k')
    ax.set_ylim(ymin=1.05*np.min((logliks_em, logliks_da)))
    fig1.savefig('./logliks_byiter.pdf')

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(times_em, logliks_em)
    ax.plot(times_da, logliks_da)
    ax.plot(times_sa, logliks_sa)
    ax.axhline(y=data_loglik, color='k')
    ax.set_ylim(ymin=1.05*np.min((logliks_em, logliks_da)))
    ax.set_xlim(xmax=min(np.max(times_em), np.max(times_da)))
    fig2.savefig('./logliks_bytime.pdf')
