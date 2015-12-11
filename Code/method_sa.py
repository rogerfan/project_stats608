import time
import numpy as np
from scipy.stats import multivariate_normal as mnorm



def mm_sa(data, init_mu, init_sigma, maxIter=100):
    
    # initialize number of groups, observations and p
    num_groups = len(init_mu)
    n = len(data)
    try:
        p = len(init_mu[0])
    except TypeError:
        p = 1

    # initialize parameters for each mixture component
    curr_mu = init_mu
    curr_sigma = init_sigma
    curr_mix = init_mix

    # randomly assign classes to begin
    cls = np.random.randint(0, num_groups, n)
    # create an array to later select new classes from
    gps = np.arange(num_groups)
    # set max number of iterations
    nMax = maxIter
    nRep = 0
    
    # create cooling/weighting vector
    beta = np.array([.1] * n)
    print(len(beta))
    # create empty vector for log-likeihoods
    logLik_vect = np.empty(nMax)
    # calculate log likelihood for each observation given current 
    # class assignments
    curr_lik = calc_loglik_full(data, curr_mu, curr_sigma, cls)
    # calculate current log-likelihood for entire data
    run_loglik = np.sum(curr_lik)
    # array to select pivots from
    ids = np.arange(n)

    # initial mix
    mix = np.ones(3) / 3.0

    while nRep < nMax:

        # calculate total weight for weighted selection of pivot
        tot_weight = np.sum(beta)
        pivot = np.random.choice(ids, 1, p=beta/tot_weight)
        old_class = cls[pivot]
        # select one class to switch to
        other_gps = np.setdiff1d(gps, old_class)
        probs = mix[other_gps] / np.sum(mix[other_gps])
        cls_change = int(np.random.choice(other_gps, p=probs, size=1))
        # calculate the new likelihood under new class assignment 
        new_lik = calc_loglik_point(data[pivot,:], 
                                    curr_mu[cls_change],
                                    curr_sigma[cls_change])

        # if our new likelihood is better; accept, update LL, class assignment
        # and estimates of mu_k, sigma_k for all k
        if (new_lik >= curr_lik[pivot]):
            run_loglik = run_loglik + new_lik - curr_lik[pivot]
            curr_lik[pivot] = new_lik
            cls[pivot] = cls_change
            for k in range(num_groups):
                gp_memb = np.where(cls == k, 1, 0)
                curr_mu[k] = np.average(data, weights=gp_memb, axis=0)
                curr_sigma[k] = np.cov(data, rowvar=0, aweights=gp_memb, ddof=0)
            logLik_vect[nRep] = run_loglik
            mix[old_class] = mix[old_class] - 1.0 / n
            mix[cls_change] = mix[cls_change] + 1.0 / n
        # otherwise, we'll switch if runif < beta[pivot];
        # regardless of weather we switch, we "cool" that observation's beta
        else:
            pChange = np.random.uniform(0,1,size=1)
            if pChange < beta[pivot]:
                run_loglik = run_loglik + new_lik - curr_lik[pivot]
                curr_lik[pivot] = new_lik
                cls[pivot] = cls_change
                for k in range(num_groups):
                    gp_memb = np.where(cls == k, 1, 0)
                    curr_mu[k] = np.average(data, weights=gp_memb, axis=0)
                    curr_sigma[k] = np.cov(data, rowvar=0, aweights=gp_memb, ddof=0)
                logLik_vect[nRep] = run_loglik
                mix[old_class] = mix[old_class] - 1.0 / n
                mix[cls_change] = mix[cls_change] + 1.0 / n
            else:
                logLik_vect[nRep] = run_loglik
            beta[pivot] = beta[pivot] / 2
        nRep += 1
    print(mix)
    print([np.mean(np.where(cls == k, 1, 0)) for k in range(3)])
    return (curr_mu, curr_sigma), logLik_vect, beta



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
    num_groups = len(mus)
    n_samp = len(data)
    pdfs = np.zeros(n_samp)
    for j in range(n_samp):
        curr_cls = int(lab[j])
        pdfs[j] = mnorm.pdf(data[j,:], mus[curr_cls], sigmas[curr_cls])
    return np.log(pdfs)

def calc_loglik_point(obs, mu, sigma):
    '''
    Calculates pdf of single observation for new class

    Parameters
    ----------
    obs : observation whose likelihood is calculated given mu, sigma
    mu : mean vector for mvn
    sigma : cov. matrix for mvn

    Returns
    -------
    log-likelihood of observation for new class

    '''

    lik = mnorm.pdf(obs, mu, sigma)
    return np.log(lik)

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

    z_ind = np.zeros(n)
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
    start = time.time()
    res, logliks, beta = mm_sa(
        x, init_mu, init_sigma, maxIter=50000)
    end = time.time()
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

    curr_mu = res[0]; curr_sigma = res[1]
    data_loglik = np.sum(calc_loglik_full(x, curr_mu, curr_sigma, z_ind))
    true_loglik = np.sum(calc_loglik_full(x, mu, sigma, z_ind))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(logliks)
    #ax.axhline(y=data_loglik, color='green')
    # ax.axhline(y=true_loglik, color='red')
    # fig.savefig('./logliks_sa2.pdf')
    print("\nBest Likelihood is: ", np.max(logliks))
    print("Difference between true and estimate is: ", np.abs(logliks[-1]-true_loglik))
