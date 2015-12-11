import numpy as np
from scipy.stats import multivariate_normal as mnorm


def calc_probs(pdfs, mix, b):
    weighted_pdfs = (mix*pdfs)**b
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


def sample_multinomials(probs):
    temp = np.zeros((probs.shape[0], probs.shape[1]+1))
    temp[:,1:] = np.cumsum(probs, axis=1)
    u = np.random.uniform(size=len(probs))

    res = np.zeros(probs.shape, dtype=int)
    for i in range(probs.shape[1]):
        res[:,i] = np.logical_and(temp[:,i] <= u, u < temp[:,i+1])
    return res

