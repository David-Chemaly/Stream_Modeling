import os
import numpy as np
import pickle
from scipy.stats import truncnorm

from mock_data import get_mock_data
from priors import *
from likelihoods import *
from models_orbits import *
from models_streams import *
from dynesty_fit import dynesty_fit
from population import MCMC_fit
from corner_plots import plot_corners

def main(data_info, model_info, id, seed, q_true, nlive, dir_save, ground_truth='orbit_to_orbit', labels='orbit'):
    print('Generating Data for Process', id+1)
    dict_data, params_data, spline_data, theta_track_data = get_mock_data(q_true, 
                                                                            data_info['model'],
                                                                            data_info['prior_transform'],
                                                                            data_info['model_spline'],
                                                                            data_info['ndim'],
                                                                            seed=seed, 
                                                                            sigma=data_info['sigma'])

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params_data)

    print('Started Fit for Process', id+1)
    dict_result = dynesty_fit(dict_data, 
                    model_info['model'],
                    model_info['model_spline'],
                    model_info['log_likelihood'],
                    model_info['prior_transform'],
                    ndim=model_info['ndim'], nlive=nlive)

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)
        
    if ground_truth == 'orbit_to_orbit':
        true = params_data
    elif ground_truth == 'stream_to_stream':
        true = params_data
    elif ground_truth == 'orbit_to_stream':
        true = np.concatenate([params_data[:6], np.zeros(7)+np.nan])
    
    if labels == 'orbit':
        labels = [r'logM$_{halo}$', r'R_s', r'$q$', r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$', 
                                        r'x$_0$', r'y$_0$', r'z$_0$', r'v', r'$\hat{v_x}$', r'$\hat{v_y}$', r'$\hat{v_z}$']
    elif labels == 'stream':
        labels = [r'logM$_{halo}$', r'R_s', r'$q$', r'$\hat{x}$', r'$\hat{y}$', r'$\hat{z}$', r'logm$_{prog}$', r'r_s'
                                        r'x$_0$', r'y$_0$', r'z$_0$', r'v', r'$\hat{v_x}$', r'$\hat{v_y}$', r'$\hat{v_z}$', 'time']

    plot_corners(dict_result, model_info['model'], dict_data, true, labels)

if __name__ == '__main__':
    # Hyperparameters
    q_mean, q_sig, nlive, sigma, N, seed = 1.0, 0.1, 100, 5, 1, 41
    PATH_SAVE  = f'/data/dc824-2/test_orbit_to_orbit/q{q_mean}_qsig{q_sig}_seed{seed}_nlive{nlive}_sigma{sigma}'
    ground_truth = 'orbit_to_orbit'
    labels='orbit'
    data_type = 'orbit'
    model_type = 'orbit'

    # Data
    if data_type == 'stream':
        data_info = {
            'model': gala_stream_model_ndim16,
            'prior_transform': stream_large_prior_transform_ndim16,
            'model_spline': stream_spline,
            'ndim': 16,
            'sigma': 5,
        }
    elif data_type == 'orbit':
        data_info = {
            'model': gala_orbit_model_ndim13,
            'prior_transform': orbit_unrestricted_prior_transform_ndim13,
            'model_spline': orbit_spline,
            'ndim': 13,
            'sigma': 5,
        }

    # Model
    if model_type == 'stream':
        model_info = {
            'model': gala_stream_model_ndim16,
            'prior_transform': stream_large_prior_transform_ndim16,
            'model_spline': stream_spline,
            'log_likelihood': model_log_likelihood,
            'ndim': 16,
        }
    elif model_type == 'orbit':
        model_info = {
            'model': gala_orbit_model_ndim13,
            'prior_transform': orbit_unrestricted_prior_transform_ndim13,
            'model_spline': orbit_spline,
            'log_likelihood': model_log_likelihood,
            'ndim': 13
        }

    # Create directory if it doesn't exist
    os.makedirs(PATH_SAVE, exist_ok=True)

    # Save data_info and model_info as pickle files
    with open(f'{PATH_SAVE}/data_info.pkl', 'wb') as f:
        pickle.dump(data_info, f)

    with open(f'{PATH_SAVE}/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

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
        main(data_info, model_info, id, s, q_true[id], nlive, PATH_SAVE, ground_truth, labels)

    N = [N//4, N//2, N]
    for n in N:
        MCMC_fit(PATH_SAVE, n, ndim=2, nwalkers=10, nsteps=1000)

    with open(f'{PATH_SAVE}/dict_q.pkl', 'rb') as file:
        dict_q= pickle.load(file)

    average_q = 0
    all_q = []
    for i in range(len(dict_q)):
        average_q += dict_q[i].mean()
        all_q.extend(dict_q[i])
    average_q /= len(dict_q)

    plt.figure()
    plt.hist(all_q, bins=30, color='blue', histtype='bar', ec='black')
    plt.axvline(average_q, color='red', lw=2, label='Mean', linestyle='--')
    plt.axvline(q_mean, color='red', lw=2, label='True')
    plt.xlabel('q')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.savefig(f'{PATH_SAVE}/q_hist.pdf')