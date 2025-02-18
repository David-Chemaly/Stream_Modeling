import numpy as np
import emcee
import multiprocessing

import pickle
import numpy as np
from tqdm import tqdm

def combine_data(path_save, N):
    dict_flat, dict_samps, dict_params = {}, {}, {}
    all_true_params = []
    max_length = 0
    N_real = 0
    for i in tqdm( range(N), leave=True):

        try:
            with open(f'{path_save}/xx_{i+1:03d}/dict_result.pkl', 'rb') as file:
                dns = pickle.load(file)

            true_params = np.loadtxt(f'{path_save}/xx_{i+1:03d}/params.txt')
            all_true_params.append(true_params[2])

            posterior_samples = dns['samps'][:,2]

            dict_samps[i]  = dns#['samps']
            dict_params[i] = true_params

            dict_flat[i] = posterior_samples

            max_length = np.max([max_length, len(posterior_samples)])
            N_real += 1
        except:
            print(f'Failed for i={i}')

    samples_array = np.zeros([N_real, max_length])
    for i in range(N):
        try:
            samples_array[i, :len(dict_flat[i])] = dict_flat[i]
        except:
            print(f'Failed for i={i}')
    
    with open(f'{path_save}/dict_q.pkl', 'wb') as f:
        pickle.dump(dict_flat, f)

    return dict_flat

def MCMC_fit(path_save, N, ndim=2, nwalkers=10, nsteps=1000):


    dict_data = combine_data(path_save, N)

    # Initial positions of the walkers
    p0 = 2*np.random.uniform(size=(nwalkers, ndim))


    # Set up the multiprocessing pool
    with multiprocessing.Pool() as pool:
        # Set up the MCMC sampler with the pool for parallel processing
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(dict_data, ), pool=pool)
        
        # Run MCMC
        sampler.run_mcmc(p0, nsteps, progress=True)

    # Get the chain of samples after burn-in (e.g., first 1000 steps)
    samples = sampler.get_chain(discard=200, thin=15, flat=True)

    np.save(path_save+f'/population_samples_N{N}.npy', samples)

    return samples

def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

# Define the log-posterior function
def log_posterior(theta, dict_data):
    mu, sigma = theta
    
    # Prior: assuming broad uniform priors
    if mu < 0:
        return -np.inf
    elif mu > 2:
        return -np.inf
    elif sigma < 0: 
        return -np.inf
    elif sigma > 2:
        return -np.inf
    
    else:
        # Log-prior (uniform priors, log of 1 is 0, so we can ignore it)
        log_prior = 0
        
        log_likelihood = 0
        for i in dict_data.keys():
        # likelihood      = np.sum( gaussian(dict_data, mu, sigma) * np.int8(dict_data!=0), axis=1 ) / NN
            likelihood       = np.mean( gaussian(dict_data[i], mu, sigma) )
            log_likelihood  += np.log(likelihood)

        return log_prior + log_likelihood
    