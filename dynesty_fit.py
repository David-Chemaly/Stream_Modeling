import os
import numpy as np
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

def dynesty_fit(dict_data, 
                model,
                spline_model,
                log_likelihood,
                prior_transform,
                ndim=13, nlive=4000, theta_initial=0, pieces='both'):
    
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, model, spline_model, theta_initial, pieces),
                                nlive=nlive,
                                sample='rslice',  
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)
    
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }

if __name__ == '__main__':
    from mock_data import *
    from priors import *
    from models_streams import *
    from utils import *
    from likelihoods import *

    q_true = 1.
    model = agama_stream_model_ndim16
    prior_trainsform = stream_large_prior_transform_ndim16
    model_spline = get_spline_for_stream
    log_likelihood = model_log_likelihood
    ndim = 16

    dict_data, params, spline, theta_track = get_mock_data(q_true, 
                                            model,
                                            prior_trainsform,
                                            model_spline,
                                            ndim,
                                            pieces='leading',
                                            seed=1, sigma=10)
    

    dict_result = dynesty_fit(dict_data, 
                    agama_stream_model_ndim15,
                    model_spline,
                    log_likelihood,
                    stream_large_prior_transform_ndim15,
                    ndim=15, nlive=1000, theta_initial=dict_data['theta'][0], pieces='leading')

    with open(f'/data/dc824-2/test_S2S.pkl', 'wb') as f:
        pickle.dump(dict_result, f)