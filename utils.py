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

def order_points_nearest_neighbor(x, y, x_start, y_start, max_dist=1000):
    """
    Orders the given (x, y) points using a nearest neighbor approach, 
    starting from a custom (x_start, y_start) point.
    
    Returns:
        ordered_indices: List of indices that sort the original points
    """
    # Combine x and y into coordinate pairs
    points = np.column_stack((x, y))
    
    # Find the index of the nearest point to the given start point
    distances = np.linalg.norm(points - np.array([x_start, y_start]), axis=1)
    start_idx = np.argmin(distances)  # Closest point as start

    # Initialize ordering
    ordered_indices = [start_idx]
    remaining_indices = set(range(len(points))) - {start_idx}

    # Order points using nearest neighbor approach
    while remaining_indices:
        last_idx = ordered_indices[-1]
        last_point = points[last_idx]
        
        # Find nearest remaining neighbor
        remaining_points = points[list(remaining_indices)]
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] > max_dist:
            break
        
        # Convert back to the original index
        original_idx = list(remaining_indices)[nearest_idx]
        ordered_indices.append(original_idx)
        remaining_indices.remove(original_idx)

    return ordered_indices

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

def get_spline_for_stream(first_theta, first_r, theta_prog, r_prog, second_theta, second_r):
    first_theta = np.unwrap(np.flip(first_theta))
    first_r = np.flip(first_r)

    NN = first_theta[-1] // (2*np.pi)

    correct_theta_prog = theta_prog + 2*np.pi*NN

    first_r = first_r[first_theta < correct_theta_prog]
    first_theta = first_theta[first_theta < correct_theta_prog]

    second_theta = np.unwrap(second_theta + 2*np.pi*NN)

    seconnd_r = second_r[second_theta > correct_theta_prog]
    second_theta = second_theta[second_theta > correct_theta_prog]
    theta_track = np.concatenate((first_theta, np.array([correct_theta_prog]), second_theta))
    r_track     = np.concatenate((first_r, np.array([r_prog]), seconnd_r))
    
    if (np.diff(theta_track) > 0).all():
        spline = CubicSpline(theta_track, r_track, extrapolate=False)
    else:
        spline = None
        theta_track = None

    return spline, theta_track
    
def restriction(theta, r, theta_min = 3*np.pi/2, theta_max = 6*np.pi, r_min = 10, dr_max = 500):
    if (np.diff(theta) > 0).all() & (theta.ptp() > theta_min) & (r.min() > r_min) & (theta.ptp() < theta_max)   & (r.ptp() < dr_max):
        return True
    else:
        return False
    
def compute_gamma(all_xyz_stream, xyz_prog):

    all_r_stream = np.linalg.norm(all_xyz_stream, axis=-1)
    r_prog = np.linalg.norm(xyz_prog, axis=-1)
    gamma = np.sum(all_r_stream - r_prog[None], axis=-1)
    gamma_min = np.min(gamma)
    gamma_max = np.max(gamma)
    gamma = 2 * (gamma - gamma_min) / (gamma_max - gamma_min) - 1

    return gamma