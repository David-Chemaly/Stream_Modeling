import numpy as np

from priors import *
from utils import *

def get_mock_data(q_true, 
                    model,
                    prior_trainsform,
                    model_spline,
                    ndim,
                    seed=42, sigma=10, n_ang=72, min_particle=3, max_dist=80):
    
    rng = np.random.RandomState(seed)
    correct = False

    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array( prior_trainsform(p) )
        params[2] = q_true

        xyz_model, xyz_prog = model(params, n_steps=int(1e3))

        spline, theta_track = model_spline(xyz_model)

        if spline is not None:
            theta_new = np.linspace(theta_track.min(), theta_track.max(), 1000)
            r_new = spline(theta_new)

            if restriction(theta_new, r_new):
                correct = True

    theta_bin = np.arange(theta_track.min(), theta_track.max(), 18 * np.pi/180) #np.arange(0, 360, 360/n_ang) * np.pi/180
    r_bin     = spline(theta_bin)

    arg_in = ~np.isnan(r_bin)

    theta_data = theta_bin[arg_in]
    r_data     = r_bin[arg_in]
    x_data = r_data * np.cos(theta_data)
    y_data = r_data * np.sin(theta_data)
    
    if sigma == 0:
        r_sig = 0
        noise = 0
    else:
        r_sig = r_data*sigma/100
        noise = rng.normal(0, r_sig)

    dict_data = {'theta': theta_data, 'r': r_data+noise, 'x': x_data + noise*np.cos(theta_data), 'y': y_data + noise*np.sin(theta_data), 'r_sig': r_sig, 'noise': noise}

    return dict_data, params, spline, theta_track