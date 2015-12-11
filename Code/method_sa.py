from time import process_time
import numpy as np
from scipy.stats import multivariate_normal as mnorm



def mm_sa(data, init_mu, init_sigma, num_iter=100):
    # initialize number of groups, observations and p
    num_groups = len(init_mu)
    n = len(data)
    try:
        p = data.shape[1]
    except IndexError:
        p = 1

    # Randomly assign classes to begin
    curr_class = np.random.randint(0, num_groups, n)
    groups = np.arange(num_groups)  # list of possible groups
    ids = np.arange(n)  # array to select pivots from

    # Initialize curr parameters based on assigned classes
    # curr_mu = np.array(
    #     [np.mean(data[curr_class == k], axis=0) for k in range(num_groups)])
    # curr_sigma = np.array(
    #     [np.cov(data[curr_class == k], rowvar=0) for k in range(num_groups)])

    curr_mu = init_mu.copy()
    curr_sigma = init_sigma.copy()
    curr_mix = np.array([np.sum(curr_class == k)/n for k in range(num_groups)])

    beta = 0.1*np.ones(n)  # cooling/weighting vector

    loglik_vect = np.zeros(num_iter)  # for storing logliks at each iter
    time_iter = np.zeros(num_iter)

    curr_logliks = calc_loglik_full(data, curr_mu, curr_sigma, curr_class)
    tot_loglik = np.sum(curr_logliks)

    for iternum in range(num_iter):
        # Select pivot by randomly using beta as weights
        pivot = np.random.choice(ids, 1, p=beta/np.sum(beta))
        old_class = curr_class[pivot]

        # Select a class to switch to
        other_groups = np.setdiff1d(groups, old_class)
        probs = curr_mix[other_groups] / np.sum(curr_mix[other_groups])
        cls_change = int(np.random.choice(other_groups, p=probs, size=1))

        # calculate the new likelihood under new class assignment
        new_loglik = mnorm.logpdf(
            data[pivot,:], curr_mu[cls_change], curr_sigma[cls_change])

        # if our new likelihood is better; accept, update LL, class assignment
        # and estimates of mu_k, sigma_k for all k
        pchange = np.random.uniform()
        if new_loglik >= curr_logliks[pivot] or pchange < beta[pivot]:
            tot_loglik = tot_loglik + new_loglik - curr_logliks[pivot]
            curr_logliks[pivot] = new_loglik
            curr_class[pivot] = cls_change
            for k in range(num_groups):
                curr_mu[k] = np.mean(data[curr_class == k], axis=0)
                curr_sigma[k] = np.cov(data[curr_class == k], rowvar=0)
            loglik_vect[iternum] = tot_loglik
            curr_mix[old_class] = curr_mix[old_class] - 1.0 / n
            curr_mix[cls_change] = curr_mix[cls_change] + 1.0 / n

        if new_loglik < curr_logliks[pivot]:
            beta[pivot] *= 0.5  # Cool that obs beta regardless of switching

        loglik_vect[iternum] = tot_loglik

    print(curr_mix)
    print([np.mean(np.where(curr_class == k, 1, 0)) for k in range(3)])
    return (curr_mu, curr_sigma), loglik_vect, beta



def calc_loglik_full(data, mus, sigmas, lab):
    '''
    Calculates log-likelihood of data

    Parameters
    ----------
    data : 2-d array
    mus : 2-d array
        Array of mean vectors.
    sigmas : list of 2-d arrays or 3-d array
        Covariance matrices.

    Returns
    -------
    log-likelihood

    '''
    pdfs = np.zeros(len(data))
    for g in np.unique(lab):
        pdfs[lab == g] = mnorm.logpdf(data[lab == g], mus[g], sigmas[g])
    return pdfs


if __name__ == '__main__':
    from matplotlib import colors
    import matplotlib.pyplot as plt

    n = 2000
    # np.random.seed(29643)
    np.random.seed(100)

    # Simulate 3 groups
    num_groups = 3
    sigma = np.array(
        [[[1.0,  0.1], [ 0.1, 0.3]],
         [[1.0, -0.1], [-0.1, 1.0]],
         [[0.5, -0.5], [-0.5, 1.0]]])
    mu = np.array([[0., 2.], [2., 1.], [2., 3.]])

    mix = np.array([.15, .7, .15])

    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')
    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=np.int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2


    init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
    init_sigma = [np.identity(2) for i in range(3)]
    init_mix = np.array([1., 1., 1.])/3
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # #
    # run the algorithm
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # #
    start = process_time()
    res, logliks, beta = mm_sa(
        x, init_mu, init_sigma, num_iter=50000)
    end = process_time()
    print("\n\n")
    print("Total time elapsed: ", end - start)
    print("\n\n")
    cmap, norm = colors.from_levels_and_colors(
                levels=[0, 1, 2], colors=['magenta', 'cyan', 'green'], extend='max')

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(x[:,0], x[:,1], c=z_ind, cmap=cmap, norm=norm)
    # fig.savefig('./sim_data_sa2.pdf')

    n, bins, patches = plt.hist(beta, 100, facecolor='green', alpha=0.75)

    plt.xlabel('Cooling Parameter')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('dist_weights.png')

    # add a 'best fit' line
    l = plt.plot(bins)


    data_mu = np.array(
        [np.mean(x[z_ind == k], axis=0) for k in range(num_groups)])
    data_sigma = np.array(
        [np.cov(x[z_ind == k], rowvar=0) for k in range(num_groups)])

    data_loglik = np.sum(calc_loglik_full(x, data_mu, data_sigma, z_ind))
    true_loglik = np.sum(calc_loglik_full(x, mu, sigma, z_ind))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(logliks)
    ax.axhline(y=data_loglik, color='green')
    ax.axhline(y=true_loglik, color='red')
    fig.savefig('./logliks_sa2.pdf')
    print("\nBest Likelihood is: ", np.max(logliks))
    print("Difference between true and estimate is: ", np.abs(logliks[-1]-true_loglik))
