from time import process_time

import numpy as np
from scipy.stats import multivariate_normal as mnorm
import matplotlib.pyplot as plt
from matplotlib import colors

from methods import em_alg, sim_anneal
from utilities import calc_pdfs, calc_loglik


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

np.savetxt(
    './intermediate_data/sim_data.csv',
    np.column_stack((x, z_ind)),
    header='x1, x2, z', delimiter=',', comments='')

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
    x, init_mu, init_sigma, init_mix, num_iter=250, seed=29624,
    temp_func=lambda i: max(1e-4, 100*.992**i))

# Save results
colnames = \
    'logliks_em, times_em, logliks_da, times_da, logliks_sa, times_sa'
np.savetxt(
    './intermediate_data/singlerun_results.csv',
    np.column_stack((logliks_em, times_em,
                     logliks_da, times_da,
                     logliks_sa['best'], times_sa)),
    header=colnames, delimiter=',', comments='')

np.savetxt('./intermediate_data/sa_singlerun.csv',
    np.column_stack((logliks_sa['curr'], logliks_sa['best'])),
    header='logliks_curr, logliks_best', delimiter=',', comments='')

# Run SA multiple times
n_runs = 100
np.random.seed(1913463)
for t in (950, 975, 992, 999):
    print(t)
    run_storage = np.zeros((251*n_runs, 4))
    for i in range(n_runs):
        _, (loglik, time) = sim_anneal(
            x, init_mu, init_sigma, init_mix, num_iter=250,
            temp_func=lambda i: max(1e-4, 100*(t/1000)**i))
        run_storage[i*251:(i+1)*251, 0] = i
        run_storage[i*251:(i+1)*251, 1] = np.arange(251)
        run_storage[i*251:(i+1)*251, 2] = loglik['curr']
        run_storage[i*251:(i+1)*251, 3] = loglik['best']

    np.savetxt(
        './intermediate_data/sa_t{}.csv'.format(t), run_storage,
        header='run, iter, loglik_curr, loglik_best',
        delimiter=',', comments='')

# Calculate true loglik
data_mu = np.array(
    [np.mean(x[z_ind == k], axis=0) for k in range(num_groups)])
data_sigma = np.array(
    [np.cov(x[z_ind == k], rowvar=0) for k in range(num_groups)])

data_loglik = calc_loglik(
    x, calc_pdfs(x, data_mu, data_sigma), z)

print('True loglik: {}'.format(data_loglik))
# -5196.976212643527
