import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal as mnorm

from methods import em_alg, sim_anneal

# Setup Parameters and Initial Values
num_groups = 3
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
# np.random.seed(8234623)
for n in (500, 1000, 2000):
    print("Simulations for n={}.".format(n))
    run_storage_em = np.zeros(((iter_num+1)*n_runs, 4))
    run_storage_da = np.zeros(((iter_num+1)*n_runs, 4))
    run_storage_sa = np.zeros(((iter_num + 1)*n_runs, 5))
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
        try:
            _, (loglik_sa, time_sa) = sim_anneal(
                x, init_mu, init_sigma, init_mix, num_iter=iter_num,
                temp_func=lambda i: max(1e-4, 100*.992**i))
        except LinAlgError:
            try:
                _, (loglik_sa, time_sa) = sim_anneal(
                    x, init_mu, init_sigma, init_mix, num_iter=iter_num,
                    temp_func=lambda i: max(1e-4, 100*.992**i))
            except LinAlgError:
               _, (loglik_sa, time_sa) = sim_anneal(
                   x, init_mu, init_sigma, init_mix, num_iter=iter_num,
                   temp_func=lambda i: max(1e-4, 100*.992**i))

        # Run EM
        _, (logliks_em, times_em) = em_alg(
        x, init_mu, init_sigma, init_mix, num_iter=iter_num)

        # Run DA
        _, (logliks_da, times_da) = em_alg(
            x, init_mu, init_sigma, init_mix, num_iter=iter_num,
            beta_func=lambda i: 1.-np.exp(-(i+1)/5))

        # Store results
        stor_size = iter_num+1
        run_storage_em[j*stor_size:(j+1)*stor_size, 0] = j
        run_storage_em[j*stor_size:(j+1)*stor_size, 1] = np.arange(stor_size)
        run_storage_em[j*stor_size:(j+1)*stor_size, 2] = logliks_em
        run_storage_em[j*stor_size:(j+1)*stor_size, 3] = times_em

        run_storage_da[j*stor_size:(j+1)*stor_size, 0] = j
        run_storage_da[j*stor_size:(j+1)*stor_size, 1] = np.arange(stor_size)
        run_storage_da[j*stor_size:(j+1)*stor_size, 2] = logliks_da
        run_storage_da[j*stor_size:(j+1)*stor_size, 3] = times_da

        run_storage_sa[j*stor_size:(j+1)*stor_size, 0] = j
        run_storage_sa[j*stor_size:(j+1)*stor_size, 1] = np.arange(stor_size)
        run_storage_sa[j*stor_size:(j+1)*stor_size, 2] = loglik_sa['curr']
        run_storage_sa[j*stor_size:(j+1)*stor_size, 3] = loglik_sa['best']
        run_storage_sa[j*stor_size:(j+1)*stor_size, 4] = time_sa

    np.savetxt(
        './intermediate_data/em_{}.csv'.format(n), run_storage_em,
        header='run, iter, loglik_curr, time',
        delimiter=',', comments='')
    np.savetxt(
        './intermediate_data/da_{}.csv'.format(n), run_storage_da,
        header='run, iter, loglik_curr, time',
        delimiter=',', comments='')
    np.savetxt(
        './intermediate_data/sa_{}.csv'.format(n), run_storage_sa,
        header='run, iter, loglik_curr, loglik_best, time',
        delimiter=',', comments='')
