
import numpy as np
from scipy.spatial.transform import Rotation as R

import agama
agama.setUnits(length=1, velocity=1, mass=1)

from astropy import units as auni
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline


from utils import *
from agama_spray import create_stream_particle_spray_with_progenitor

def agama_stream_model_ndim16(params):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    logv, dirvx, dirvy, dirvz, \
    time_total = params

    rot_mat = get_mat(dirx, diry, dirz)
    rot = R.from_matrix(rot_mat)
    euler_angles = rot.as_euler('xyz', degrees=False)

    densitynorm = compute_densitynorm(10**logM, Rs, q)
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, 
                                axisRatioY=1, axisRatioZ=q, orientation=euler_angles)

    num_particles = int(1e4)  # number of particles in the stream

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)
    vx = 10**logv * v_dir[0]
    vy = 10**logv * v_dir[1]
    vz = 10**logv * v_dir[2]

    posvel_sat = np.array([pos_init_x, pos_init_y, pos_init_z, vx, vy, vz])

    xyz_stream, gamma, xyz_prog = create_stream_particle_spray_with_progenitor(time_total, num_particles, pot_host, posvel_sat, 10**logm, rs, gala_modified=True)

    return xyz_stream, gamma, xyz_prog

def agama_stream_model_ndim15(params, theta_initial=0):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_z, \
    logv, dirvx, dirvy, dirvz, \
    time_total = params

    rot_mat = get_mat(dirx, diry, dirz)
    rot = R.from_matrix(rot_mat)
    euler_angles = rot.as_euler('xyz', degrees=False)

    densitynorm = compute_densitynorm(10**logM, Rs, q)
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, 
                                axisRatioY=1, axisRatioZ=q, orientation=euler_angles)

    num_particles = int(1e4)  # number of particles in the stream

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)
    vx = 10**logv * v_dir[0]
    vy = 10**logv * v_dir[1]
    vz = 10**logv * v_dir[2]

    x_rot = pos_init_x * np.cos(theta_initial) 
    y_rot = pos_init_x * np.sin(theta_initial) 
    posvel_sat = np.array([x_rot, y_rot, pos_init_z, vx, vy, vz])

    xyz_stream, gamma, xyz_prog = create_stream_particle_spray_with_progenitor(time_total, num_particles, pot_host, posvel_sat, 10**logm, rs, gala_modified=True)

    return xyz_stream, gamma, xyz_prog

def gala_stream_model_ndim16(params, dt=-10):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    logv, dirvx, dirvy, dirvz, \
    t_end = params

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)
    vx = 10**logv * v_dir[0]
    vy = 10**logv * v_dir[1]
    vz = 10**logv * v_dir[2]
    
    w0 = gd.PhaseSpacePosition(
        pos=np.array([pos_init_x, pos_init_y, pos_init_z]) * auni.kpc,
        vel=np.array([vx, vy, vz]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    H = gp.Hamiltonian(pot)

    df = ms.FardalStreamDF(gala_modified=True, random_state=np.random.RandomState(42))

    prog_pot = gp.PlummerPotential(m=10**logm, b=rs, units=units)
    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, prog = gen.run(w0, 10**logm * auni.Msun, dt=dt* auni.Myr, n_steps=int(t_end * auni.Gyr/ abs(dt* auni.Myr)))
    xyz_stream = stream.xyz.value.T
    xyz_prog = prog.xyz.value[:, 0]

    return xyz_stream, xyz_prog

def stream_spline(xyz_stream, xyz_prog, n_theta=72, min_particle=3, max_dist=80):

    arg_lead  = np.arange(1, len(xyz_stream), 2)
    arg_trail = np.arange(0, len(xyz_stream), 2)

    x_prog = xyz_prog[0]
    y_prog = xyz_prog[1]
    r_prog = np.sqrt(x_prog**2 + y_prog**2)
    theta_prog = np.arctan2(y_prog, x_prog)
    if theta_prog < 0:
        theta_prog += 2*np.pi

    # Make sure progenitor not next to 0 line
    if 35/18*np.pi < theta_prog < 1/18*np.pi:
        spline = None
        theta_track = None
    
    else:
        # Get track from each arm assume no overlap for a given arm
        theta_trail, r_trail, x_trail, y_trail = get_track_from_theta(xyz_stream[arg_trail], n_theta=n_theta, min_particle=min_particle)
        theta_lead, r_lead, x_lead, y_lead     = get_track_from_theta(xyz_stream[arg_lead], n_theta=n_theta, min_particle=min_particle)

        # Do nearest neighbor for ordering
        trail_ordered_indices = order_points_nearest_neighbor(x_trail, y_trail, x_prog, y_prog, max_dist=max_dist)
        lead_ordered_indices  = order_points_nearest_neighbor(x_lead, y_lead, x_prog, y_prog, max_dist=max_dist)

        theta_trail_ordered = theta_trail[trail_ordered_indices]
        theta_lead_ordered  = theta_lead[lead_ordered_indices]

        r_trail_ordered = r_trail[trail_ordered_indices]
        r_lead_ordered  = r_lead[lead_ordered_indices]

        # If both arms have at least 2 points, compute curvature
        # if (len(trail_ordered_indices) < 2) and (len(lead_ordered_indices) < 2):
        #     spline = None
        #     theta_track = None

        # elif (len(trail_ordered_indices) >= 2) and (len(lead_ordered_indices) < 2):
        #     curvature_trail = np.sum(compute_curvature(x_trail[trail_ordered_indices], y_trail[trail_ordered_indices]))

        #     if curvature_trail < 0:
        #         spline, theta_track = get_spline_for_stream(theta_trail_ordered, r_trail_ordered, theta_prog, r_prog, np.array([]), np.array([]))
        #     elif curvature_trail > 0:
        #         spline, theta_track = get_spline_for_stream(np.array([]), np.array([]), r_prog, theta_trail_ordered, r_trail_ordered, theta_prog,)
        #     else:
        #         spline = None
        #         theta_track = None

        # elif (len(trail_ordered_indices) < 2) and (len(lead_ordered_indices) >= 2):
        #     curvature_lead = np.sum(compute_curvature(x_lead[lead_ordered_indices], y_lead[lead_ordered_indices]))

        #     if curvature_lead < 0:
        #         spline, theta_track = get_spline_for_stream(theta_lead_ordered, r_lead_ordered, theta_prog, r_prog, np.array([]), np.array([]))
        #         print(theta_track)
        #     elif curvature_lead > 0:
        #         spline, theta_track = get_spline_for_stream(np.array([]), np.array([]), r_prog, theta_lead_ordered, r_lead_ordered, theta_prog)
        #     else:
        #         spline = None
        #         theta_track = None

        if (len(trail_ordered_indices) >= 2) and (len(lead_ordered_indices) >= 2):
            curvature_trail = np.sum(compute_curvature(x_trail[trail_ordered_indices], y_trail[trail_ordered_indices]))
            curvature_lead  = np.sum(compute_curvature(x_lead[lead_ordered_indices], y_lead[lead_ordered_indices]))

            # If one arm is convex and the other is concave, start from the end of the convex arm
            if (curvature_lead < 0) and (curvature_trail > 0):
                spline, theta_track = get_spline_for_stream(theta_lead_ordered, r_lead_ordered, theta_prog, r_prog, theta_trail_ordered, r_trail_ordered)
            elif (curvature_lead > 0) and (curvature_trail < 0):
                spline, theta_track = get_spline_for_stream(theta_trail_ordered, r_trail_ordered, theta_prog, r_prog, theta_lead_ordered, r_lead_ordered)
            else:
                spline = None
                theta_track = None
        
        else:
            spline = None
            theta_track = None
    
    return spline, theta_track