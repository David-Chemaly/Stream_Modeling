import numpy as np
from scipy.interpolate import CubicSpline

BAD_VAL = -1e100

def model_log_likelihood(params, dict_data, model, spline_model):
    spline, theta_model = spline_model(model(params))

    if spline == None:
        logl = 2*BAD_VAL

    else:
        logl = get_logl_from_spline(spline, theta_model, dict_data)
    
    return logl
    
def get_logl_from_spline(spline, theta_model, dict_data):
    r_data = dict_data['r']
    theta_data = dict_data['theta']
    r_sig = dict_data['r_sig']


    if theta_model.max() < dict_data['theta'].min():
        N = np.ceil( (dict_data['theta'].min() - theta_model.max()) / (2*np.pi) )
        spline = CubicSpline(theta_model + 2*np.pi*N, spline(theta_model))
        theta_model += 2*np.pi*N
    elif theta_model.min() > dict_data['theta'].max():
        N = np.ceil( (theta_model.min() - dict_data['theta'].max()) / (2*np.pi) )
        spline = CubicSpline(theta_model - 2*np.pi*N, spline(theta_model))
        theta_model -=  2*np.pi*N

    if (theta_data.min() >= theta_model.min() ) & (theta_data.max() <= theta_model.max()):
        logl = -.5 * np.sum( ( (spline(theta_data) - r_data) / r_sig )**2 )
    else:
        logl     = BAD_VAL
        penalty  = max((np.maximum(theta_data, theta_model.min()) - theta_model.min())**2)
        penalty += max((np.minimum(theta_data, theta_model.max()) - theta_model.max())**2)
        logl = logl - np.abs(BAD_VAL) / 10000 * penalty 

    return logl