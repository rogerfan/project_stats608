
import numpy as np
from scipy.stats import multivariate_normal as mnorm


def em_gmm(data, init_mu, init_sigma, init_mix, num_iter=100, verbose=False):
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
    curr_mu : 2-d array
    curr_sigma : list of 2-d arrays or 3-d array
    curr_mix : array
        Final estimates of the mean, covariance, and mixture components.

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

    curr_mu = init_mu
    curr_sigma = init_sigma
    curr_mix = init_mix
    logliks = np.zeros(num_iter+1)

    for iternum in range(num_iter):
        if verbose:  # Status updates
            if np.isclose(iternum // 100, iternum / 100) and iternum != 0:
                print()
            if np.isclose(iternum // 10, iternum / 10) and iternum != 0:
                print('.', end='', flush=True)

        # E-step
        pdfs = calc_pdfs(data, curr_mu, curr_sigma)
        probs = _calc_probs(pdfs, curr_mix)

        logliks[iternum] = calc_loglik(data, pdfs, probs)

        # M-step
        for k in range(num_groups):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])
            curr_sigma[k] = np.cov(data, rowvar=0, aweights=probs[:,k], ddof=0)
        curr_mix = np.mean(probs, axis=0)

    logliks[-1] = calc_loglik(
        data,
        calc_pdfs(data, curr_mu, curr_sigma),
        _calc_probs(pdfs, curr_mix)
    )

    return (curr_mu, curr_sigma, curr_mix), logliks


def _calc_probs(pdfs, mix):
    weighted_pdfs = mix*pdfs
    tot_pdfs = np.sum(weighted_pdfs, axis=1)
    probs = weighted_pdfs / tot_pdfs[:,np.newaxis]
    return probs


def calc_pdfs(data, mus, sigmas):
    '''
    Calculates pdf of each observation for each group.

    Parameters
    ----------
    data : 2-d array
    mus : 2-d array
        Array of mean vectors.
    sigmas : list of 2-d arrays or 3-d array
        Covariance matrices.

    Returns
    -------
    pdfs : 2-d array
        pdfs[i, j] contains the pdf for the ith observation using the jth
        group.

    '''
    num_groups = len(mus)
    pdfs = np.zeros((len(data), num_groups))
    for k in range(num_groups):
        pdfs[:,k] = mnorm.pdf(data, mus[k], sigmas[k])
    return pdfs


def calc_loglik(data, pdfs, probs):
    '''
    Calculates the log likelihood of the data. If probs is an indicator, then
    the log likelihood of the data is
    ``log p(x) = sum_i sum_k probs_ik log p_k(x_i)''
    This motivates the weighted log-likelihood to use when probs contains
    probabilities which has the same formula.

    Parameters
    ----------
    data : 2-d array
    pdfs : 2-d array
        Array of pdfs of each observation for each group.
    probs : 2-d array
        Array of the probability that each observation is in each group.

    Returns
    -------
    loglik : float
        Log likelihood of the data.

    '''
    logpdfs = np.log(pdfs)
    loglik = np.sum(probs * logpdfs)
    return(loglik)


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

    z_ind = np.zeros(n)
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

    res, logliks = em_gmm(x, init_mu, init_sigma, init_mix, num_iter=400)

    # Plotting
    data_mu = np.array(
        [np.mean(x[z_ind == k], axis=0) for k in range(num_groups)])
    data_sigma = np.array(
        [np.cov(x[z_ind == k], rowvar=0) for k in range(num_groups)])

    data_loglik = calc_loglik(
        x, calc_pdfs(x, data_mu, data_sigma), z)
    true_loglik = calc_loglik(x, calc_pdfs(x, mu, sigma), z)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(logliks)
    ax.axhline(y=data_loglik, color='green')
    ax.axhline(y=true_loglik, color='red')
    fig.savefig('./logliks.pdf')
