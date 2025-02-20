
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


def gala_orbit_model_ndim12(params,  theta_initial=0, n_steps=int(1e3)):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    pos_init_x, pos_init_z, \
    vel, dirvx, dirvy, dirvz = params

    t_end = 2 # Gyr always

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)

    vel = 10 ** vel
    vel_init_x = vel * v_dir[0]
    vel_init_y = vel * v_dir[1]
    vel_init_z = vel * v_dir[2]
    
    x_rot = pos_init_x * np.cos(theta_initial) # - pos_init_y * np.sin(theta_initial)
    y_rot = pos_init_x * np.sin(theta_initial) # + pos_init_y * np.cos(theta_initial)
    w0 = gd.PhaseSpacePosition(
        pos=np.array([x_rot, y_rot, pos_init_z]) * auni.kpc,
        vel=np.array([vel_init_x, vel_init_y, vel_init_z]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    orbit = pot.integrate_orbit(w0,
                                dt=t_end / n_steps * auni.Gyr,
                                n_steps=n_steps)
    xout, yout, zout = orbit.x.to_value(auni.kpc), orbit.y.to_value(
        auni.kpc), orbit.z.to_value(auni.kpc)
    
    xyz_orbit = np.array([xout, yout, zout]).T
    xyz_prog  = np.array([xout[-1], yout[-1], zout[-1]]).T

    return xyz_orbit, xyz_prog

def gala_orbit_model_ndim13(params, n_steps=int(1e3)):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel, dirvx, dirvy, dirvz = params

    t_end = 2 # Gyr always

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    v_dir = np.array([dirvx, dirvy, dirvz])
    v_dir = v_dir/np.linalg.norm(v_dir)

    vel = 10 ** vel
    vel_init_x = vel * v_dir[0]
    vel_init_y = vel * v_dir[1]
    vel_init_z = vel * v_dir[2]
    
    w0 = gd.PhaseSpacePosition(
        pos=np.array([pos_init_x, 0, pos_init_z]) * auni.kpc,
        vel=np.array([vel_init_x, vel_init_y, vel_init_z]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    orbit = pot.integrate_orbit(w0,
                                dt=t_end / n_steps * auni.Gyr,
                                n_steps=n_steps)
    xout, yout, zout = orbit.x.to_value(auni.kpc), orbit.y.to_value(
        auni.kpc), orbit.z.to_value(auni.kpc)
    
    xyz_orbit = np.array([xout, yout, zout]).T
    xyz_prog  = np.array([xout[-1], yout[-1], zout[-1]]).T

    return xyz_orbit, xyz_prog

def orbit_spline(xyz_model, xyz_prog):
    x_model = xyz_model[:,0]
    y_model = xyz_model[:,1]
    r_model = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.unwrap( np.arctan2(y_model, x_model) )

    if (np.diff(theta_model) <= 0).any():
        spline = None
        theta_model = None

    else:
        if theta_model[0] < 0:
            theta_model += 2*np.pi*np.ceil(-theta_model[0]/(2*np.pi))
        spline = CubicSpline(theta_model, r_model)

    return spline, theta_model