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
                ndim=13, nlive=4000, theta_initial=0):
    
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data, model, spline_model, theta_initial),
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