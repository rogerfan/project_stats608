
import numpy as np
from scipy.stats import multivariate_normal as mnorm


def em_gmm(data, init_mu, init_sigma, init_mix, num_iter=100, verbose=False):
    if len(init_mu) != len(init_sigma) or len(init_sigma) != len(init_mix):
        raise ValueError(
            'Number of initial values needs to match number of groups.')
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

    for iternum in range(num_iter):
        if verbose:
            if np.isclose(iternum // 100, iternum / 100) and iternum != 0:
                print()
            if np.isclose(iternum // 10, iternum / 10) and iternum != 0:
                print('.', end='', flush=True)

        # E-step
        pdfs = np.zeros((n, num_groups))
        for k in range(num_groups):
            pdfs[:,k] = mnorm.pdf(data, curr_mu[k], curr_sigma[k])
        weighted_pdfs = curr_mix*pdfs
        tot_pdfs = np.sum(weighted_pdfs, axis=1)
        probs = weighted_pdfs / tot_pdfs[:,np.newaxis]

        # M-step
        for k in range(num_groups):
            curr_mu[k] = np.average(data, axis=0, weights=probs[:,k])
            curr_sigma[k] = np.cov(data, rowvar=0, aweights=probs[:,k], ddof=0)
        curr_mix = np.mean(probs, axis=0)

    return curr_mu, curr_sigma, curr_mix


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import colors

    n = 200
    np.random.seed(29643)

    # # Simulate 2 groups
    # mu = np.array([[0, 2], [2, 1]])
    # sigma = [np.array([[1, .1], [.1, .3]]),
    #          np.array([[1, -.1], [-.1, 1]])]
    # mix = np.array([.2, .8])

    # x0 = mnorm.rvs(mu[0], sigma[0], size=n)
    # x1 = mnorm.rvs(mu[1], sigma[1], size=n)
    # z = np.random.binomial(1, mix[1], size=n).astype('Bool')

    # x = x0.copy()
    # x[z] = x1[z]

    # Simulate 3 groups
    mu = np.array([[0, 2], [2, 1], [2, 3]])
    sigma = [np.array([[1, .1], [.1, .3]]),
             np.array([[1, -.1], [-.1, 1]]),
             np.array([[.5, -.5], [-.5, 1.]])]
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

    res = em_gmm(x, init_mu, init_sigma, init_mix)
