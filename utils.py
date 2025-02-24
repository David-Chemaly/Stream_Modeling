import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

def compute_densitynorm(M, Rs, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * np.pi * Rs**3) / q
    densitynorm = M / C
    return densitynorm

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()

def get_track_from_theta(xyz_stream, n_theta=36, min_particle=5):
    x_stream, y_stream = xyz_stream[:,0], xyz_stream[:,1]
    r_stream = np.sqrt(x_stream**2 + y_stream**2)
    theta_stream = np.arctan2(y_stream, x_stream)

    theta_stream[theta_stream < 0] += 2*np.pi
    theta_edges = np.linspace(0, 2*np.pi, n_theta+1)

    theta_mean, r_mean =  [], []
    for i in range(n_theta):
        arg = (theta_stream > theta_edges[i]) & (theta_stream < theta_edges[i+1])

        if np.sum(arg) >= min_particle:
            theta_mean.append( (theta_edges[i] + theta_edges[i+1]) / 2 )
            r_mean.append(np.median(r_stream[arg]))

    theta_mean = np.array(theta_mean)
    r_mean = np.array(r_mean)
    x_mean = r_mean * np.cos(theta_mean)
    y_mean = r_mean * np.sin(theta_mean)

    return theta_mean, r_mean, x_mean, y_mean

def compute_curvature(x, y):
    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula with sign
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    
    return curvature

def get_spline_for_stream(xyz_stream, gamma, xyz_prog, pieces='both', num_bins = 30, uniform = 'linear', min_particule = 13, min_track=5):
    x_stream, y_stream = xyz_stream[:, -1, 0], xyz_stream[:, -1, 1]

    if uniform == 'linear':
        # Compute the bin edges such that each bin has a constant size
        bin_edges = np.linspace(np.min(gamma), np.max(gamma), num_bins + 1)
    elif uniform == 'count':
        # Compute the bin edges such that each bin has an equal number of points
        bin_edges = np.percentile(gamma, np.linspace(0, 100, num_bins + 1))

    # Bin the gamma values
    indices = np.digitize(gamma, bin_edges)

    average_x_in_bins, average_y_in_bins, average_gamma_in_bins = [], [], []
    for i in range(1, num_bins + 1):
        if np.sum(indices == i) >= min_particule:
            average_x_in_bins.append(np.median(x_stream[indices == i]))
            average_y_in_bins.append(np.median(y_stream[indices == i]))
            average_gamma_in_bins.append(np.median(gamma[indices == i]))
    average_x = np.array(average_x_in_bins)
    average_y = np.array(average_y_in_bins)
    average_gamma = np.array(average_gamma_in_bins)

    if pieces == 'leading':
        arg_leading = average_gamma > 0
        average_gamma = average_gamma[arg_leading]
        average_x = average_x[arg_leading]
        average_y = average_y[arg_leading]
    elif pieces == 'trailing':
        arg_trailing = average_gamma < 0
        average_gamma = average_gamma[arg_trailing]
        average_x = average_x[arg_trailing]
        average_y = average_y[arg_trailing]

    if len(average_x) < min_track:
        r_spline = None
        theta_model = None
    else:
        curvature = compute_curvature(average_x, average_y).sum()

        if curvature < 0:
            average_x = average_x[::-1]
            average_y = average_y[::-1]

        average_r = np.sqrt(average_x**2 + average_y**2)
        average_theta = np.arctan2(average_y, average_x)
        average_theta[average_theta < 0] += 2*np.pi
        theta_model = np.unwrap(average_theta)


        if (np.diff(theta_model) <= 0).any():
            r_spline = None
            theta_model = None  
        else:
            r_spline = CubicSpline(theta_model, average_r)

    return r_spline, theta_model
    
    
def restriction(theta, r, theta_min = np.pi/2, theta_max = 7*np.pi/2, r_min = 10, dr_max = 500):
    if (np.diff(theta) > 0).all() & (theta.ptp() > theta_min) & (r.min() > r_min) & (theta.ptp() < theta_max)   & (r.ptp() < dr_max):
        return True
    else:
        return False
    