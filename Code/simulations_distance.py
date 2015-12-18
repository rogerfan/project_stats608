import numpy as np
from numpy.linalg import LinAlgError, norm
from scipy.stats import multivariate_normal as mnorm

from methods import em_alg, sim_anneal

def cluster_dist(true_mu, est_mu):
    ngrps = len(true_mu)
    est_mu_copy = np.copy(est_mu)
    differences = np.zeros(ngrps)
    for i in range(ngrps):
        mu = true_mu[i]
        diff = [np.linalg.norm(mu - est_mu_copy[j]) for j in range(len(est_mu_copy))]
        differences[i] = np.min(diff)
        remove = np.argmin(diff)
        est_mu_copy = np.delete(est_mu_copy, remove, axis=0)
    return(np.mean(differences))



# Setup Parameters and Initial Values
num_groups = 3
n = 2000
mu = np.array([[0., 2.], [2., 1.], [2., 3.]])
sigma = np.array(
    [[[1.0,  0.1], [ 0.1, 0.3]],
     [[1.0, -0.1], [-0.1, 1.0]],
     [[0.5, -0.5], [-0.5, 1.0]]])
mix = np.array([.15, .7, .15])

init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
init_sigma = [np.identity(2) for i in range(3)]
init_mix = np.array([1., 1., 1.])/3

n_runs = 100
iter_num = 250
np.random.seed(1913463)

run_storage = np.zeros((3*n_runs, 3))
methods = np.arange(3)
print(len(methods))
np.random.seed(100)
for j in range(n_runs):
    print("Iteration: {}".format(j))

    # Simulate Data
    xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
    z = np.random.multinomial(1, mix, size=n).astype('Bool')

    x = xs[0].copy()
    x[z[:,1]] = xs[1][z[:,1]]
    x[z[:,2]] = xs[2][z[:,2]]

    z_ind = np.zeros(n, dtype=int)
    z_ind[z[:,1]] = 1
    z_ind[z[:,2]] = 2

    # Run SA
    # Note that some runs for small sample sizes run have issues with
    # singular covariance matrices, so I simply try a couple times
    # if there is an issue.

    res_sa, _ = sim_anneal(
        x, init_mu, init_sigma, init_mix, num_iter=iter_num,
        temp_func=lambda i: max(1e-4, 100*.992**i))

    # Run EM
    res_em, _ = em_alg(
    x, init_mu, init_sigma, init_mix, num_iter=iter_num)

    # Run DA
    res_da, _ = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=iter_num,
        beta_func=lambda i: 1.-np.exp(-(i+1)/5))
    
    distances = np.array([cluster_dist(mu, res_sa[0]),
                          cluster_dist(mu, res_em[0]),
                          cluster_dist(mu, res_da[0])])
    # Store results
    run_storage[(j*3):((j+1)*3), 0] = j
    run_storage[(j*3):((j+1)*3), 1] = methods
    run_storage[(j*3):((j+1)*3), 2] = distances


np.savetxt(
    './intermediate_data/dist.csv', run_storage,
    header='run, method, dist',
    delimiter=',', comments='')
