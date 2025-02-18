import os
import numpy as np
import pickle
from scipy.stats import truncnorm

from mock_data import get_mock_data
from priors import *
from likelihoods import *
from dynesty import dynesty_fit
from population import MCMC_fit

def main(data_type, model_type, id, seed, q_true, ndim, nlive, sigma, dir_save):
    dict_data, params_data, _, _ = get_mock_data(q_true, ndim, track=data_type, seed=seed, sigma=sigma, n_ang=72, min_particle=3, max_dist=80)

    if model_type == 'stream':
        log_likelihood  = stream_log_likelihood_ndim16
        prior_transform = stream_large_prior_transform_ndim16
    elif model_type == 'orbit':
        log_likelihood = orbit_log_likelihood_ndim12
        prior_transform = orbit_large_prior_transform_ndim12

    print(log_likelihood(params_data, dict_data))

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params_data)

    dict_result = dynesty_fit(dict_data, log_likelihood, prior_transform, ndim, nlive)

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)
        

if __name__ == '__main__':
    # Hyperparameters
    q_mean, q_sig, seed, ndim, nlive, sigma, N = 1.0, 0.1, 9, 13, 4000, 5, 100    
    PATH_SAVE  = f'/data/dc824-2/simple_orbit_to_orbit/q{q_mean}_qsig{q_sig}_seed{seed}_nlive{nlive}_sigma{sigma}'
    data_type  = 'orbit'
    model_type = 'orbit'

    # Create directory if it doesn't exist
    os.makedirs(PATH_SAVE, exist_ok=True)

    # Create a list of seeds, one for each process
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e5), N)

    print(f'Running {N} processes in parallel with {os.cpu_count()} cores')

    if q_sig == 0:
        q_true = q_mean * np.zeros(N)

    else:
        trunc_gauss = truncnorm((0.5 - q_mean) / q_sig, (1.5 - q_mean) / q_sig, loc=q_mean, scale=q_sig)
        q_true      = trunc_gauss.rvs(N, random_state=rng)

    # Prepare arguments for each process
    for id, s in enumerate(seeds):
        main(data_type, model_type, id, s, q_true[id], ndim, nlive, sigma, PATH_SAVE)

    N = [N//4, N//2, N]
    for n in N:
        MCMC_fit(PATH_SAVE, n, ndim=2, nwalkers=10, nsteps=1000)