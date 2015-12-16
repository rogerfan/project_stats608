from time import process_time

import numpy as np
from scipy.stats import multivariate_normal as mnorm

from methods import em_alg

n = 2000

# Less overlapping
np.random.seed(29643)
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

init_mu = np.array([[0., 0.], [1., 1.], [2., 2.]])
init_sigma = [np.identity(2) for i in range(3)]
init_mix = np.array([1., 1., 1.])/3
niter = 500
_, (logliks_em, _) = em_alg(
    x, init_mu, init_sigma, init_mix, num_iter=niter)

i=0
run_storage = np.zeros(((niter+1)*2, 3))
run_storage[i*(niter+1):(i+1)*(niter+1), 0] = i
run_storage[i*(niter+1):(i+1)*(niter+1), 1] = np.arange(niter+1)
run_storage[i*(niter+1):(i+1)*(niter+1), 2] = logliks_em

# More overlapping
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

niter = 500
_, (logliks_em, _) = em_alg(
    x, init_mu, init_sigma, init_mix, num_iter=niter)
i=1
run_storage[i*(niter+1):(i+1)*(niter+1), 0] = i
run_storage[i*(niter+1):(i+1)*(niter+1), 1] = np.arange(niter+1)
run_storage[i*(niter+1):(i+1)*(niter+1), 2] = logliks_em

# Store results
np.savetxt(
    './intermediate_data/em_mean.csv', run_storage,
    header='mix, iter, loglik_curr',
    delimiter=',', comments='')


# Mixing comparison
mu = np.array([[0., 2.], [2., 1.], [2., 3.]])
sigma = np.array(
    [[[1.0,  0.1], [ 0.1, 0.3]],
     [[1.0, -0.1], [-0.1, 1.0]],
     [[0.5, -0.5], [-0.5, 1.0]]])

# Balanced mixing
np.random.seed(29643)
mix = np.array([4, 4.5, 3.75])
mix = mix / np.sum(mix)
xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
z = np.random.multinomial(1, mix, size=n).astype('Bool')

x = xs[0].copy()
x[z[:,1]] = xs[1][z[:,1]]
x[z[:,2]] = xs[2][z[:,2]]

z_ind = np.zeros(n, dtype=int)
z_ind[z[:,1]] = 1
z_ind[z[:,2]] = 2

niter = 500
_, (logliks_em, _) = em_alg(
    x, init_mu, init_sigma, init_mix, num_iter=niter)
i=0
run_storage = run_storage = np.zeros(((niter+1)*2, 3))
run_storage[i*(niter+1):(i+1)*(niter+1), 0] = i
run_storage[i*(niter+1):(i+1)*(niter+1), 1] = np.arange(niter+1)
run_storage[i*(niter+1):(i+1)*(niter+1), 2] = logliks_em

# Unbalanced mixing
np.random.seed(29643)
mix = [0.15, 0.7, 0.15]
xs = [mnorm.rvs(mu[k], sigma[k], size=n) for k in range(3)]
z = np.random.multinomial(1, mix, size=n).astype('Bool')

x = xs[0].copy()
x[z[:,1]] = xs[1][z[:,1]]
x[z[:,2]] = xs[2][z[:,2]]

z_ind = np.zeros(n, dtype=int)
z_ind[z[:,1]] = 1
z_ind[z[:,2]] = 2

# Estimate
niter = 500
_, (logliks_em, _) = em_alg(
    x, init_mu, init_sigma, init_mix, num_iter=niter)
i=1
run_storage[i*(niter+1):(i+1)*(niter+1), 0] = i
run_storage[i*(niter+1):(i+1)*(niter+1), 1] = np.arange(niter+1)
run_storage[i*(niter+1):(i+1)*(niter+1), 2] = logliks_em

np.savetxt(
    './intermediate_data/em_mix.csv', run_storage,
    header='mix, iter, loglik_curr',
    delimiter=',', comments='')
