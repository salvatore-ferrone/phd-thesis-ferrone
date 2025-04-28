import sympy as sp 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy import integrate


def get_moons_orbital_elements():
    """build the initial conditions of the moon from https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html """
    semi_major_axis = 0.3844 * 1e6 * u.km
    eccentricity = 0.0549
    inclination = 5.145*u.deg
    return semi_major_axis, eccentricity, inclination

def set_units():
    # pick the units
    unitL = u.au
    unitT = u.yr
    unitV = unitL / unitT
    unitM = u.Msun
    unitG=unitV**2 * unitL / unitM
    G_val=const.G.to(unitG)  
    return unitL, unitT, unitV, unitM, unitG, G_val  

def compute_moons_initial_conditions(semi_major_axis,eccentricity,inclination,G_val,Mprimary):
    """
    Set the initial moon phase to be a full moon, or at x=1,y=0,z=~0
    """
    ra = (1+eccentricity) * semi_major_axis
    va = np.sqrt(G_val*Mprimary * (2/ra - 1/semi_major_axis))
    # put them in vectors 
    position_moon = np.array([ra, 0, 0])
    velocity_moon = np.array([0, va, 0])
    # rotate the position and velocity vectors by the inclination
    rotation_matrix = np.array([[1, 0, 0],
                                 [0, np.cos(inclination), -np.sin(inclination)],
                                 [0, np.sin(inclination), np.cos(inclination)]])
    position_moon = np.dot(rotation_matrix, position_moon)
    velocity_moon = np.dot(rotation_matrix, velocity_moon)
    return position_moon, velocity_moon


def get_unscaled_kepler_tidal_tensor_func():
    r,x,y,z=sp.symbols('r x y z', real=True)
    T = -sp.Matrix(
    [1 - 3*x**2/r**2, 
     -3*x*y/r**2, 
     -3*x*z/r**2,
     -3*x*y/r**2,
     1 - 3*y**2/r**2,
     -3*y*z/r**2,
     -3*x*z/r**2,
     -3*y*z/r**2,
     1 - 3*z**2/r**2])
    unscaled_tidal_tensor=T.reshape(3,3)
    unscaled_tidal_tensor_func = sp.lambdify((r, x, y, z), unscaled_tidal_tensor, "numpy")
    return unscaled_tidal_tensor_func


def system_of_equations(t, y, G_val, Mearth, Dearth, omega, Msun, scaled_tidal_tensor_func):
    # unpack the state of the system 
    position_moon = y[:3]
    velocity_moon = y[3:]
    omega_vec = np.array([0, 0, omega])
    # get the current position of the earth
    x_earth = Dearth * np.cos(omega * t)
    y_earth = Dearth * np.sin(omega * t)
    z_earth = 0
    position_earth = np.array([x_earth, y_earth, z_earth])
    r_earth = np.linalg.norm(position_earth)
    # eval the tidal tensor at the position of the earth
    tidal_tensor = -(G_val*Msun/r_earth**3)*scaled_tidal_tensor_func(r_earth, position_earth[0], position_earth[1], position_earth[2])
    tidal_force = tidal_tensor.dot(position_moon)
    # evaluate the centrifugal force
    centrifugalforce = - np.cross(omega_vec, np.cross(omega_vec.T, position_moon.T).T)
    # evaluate the force from the earth
    moon_earth_distance = np.linalg.norm(position_moon)
    earth_force=-(G_val*Mearth / moon_earth_distance**3) * position_moon
    # get the net force 
    net_force = tidal_force + centrifugalforce + earth_force
    # stack up everything 
    ydot = np.concatenate([velocity_moon, net_force])
    return ydot

def two_body_rotating_frame_system_of_equations(t, y, G_val, Mearth, omega):
    # unpack the state of the system 
    position_moon = y[:3]
    velocity_moon = y[3:]
    omega_vec = np.array([0, 0, omega])
    # evaluate the centrifugal force
    centrifugalforce = - np.cross(omega_vec, np.cross(omega_vec.T, position_moon.T).T)
    # evaluate the force from the earth
    moon_earth_distance = np.linalg.norm(position_moon)
    earth_force=-(G_val*Mearth / moon_earth_distance**3) * position_moon
    # get the net force 
    net_force =  earth_force + centrifugalforce
    # stack up everything 
    ydot = np.concatenate([velocity_moon, net_force])
    return ydot


def solve_moons_equation_of_motion(
        t_span = (0, 4),
        rtol=1e-10,
        atol=1e-10,
        t_eval_n_points = 1000,
        method='RK45'
    ):
    
    # get the proper units 
    unitL, unitT, unitV, unitM, unitG, G_val = set_units()
    # get the earth and sun 
    Dearth = 1*unitL
    omega = 2*np.pi/(1*unitT) 
    Msun = const.Msun.to(unitM).value
    Mearth = const.Mearth.to(unitM).value
    Dearth = Dearth.to(unitL).value
    omega = omega.value
    # get the moons orbital elements 
    semi_major_axis, eccentricity, inclination = get_moons_orbital_elements()
    # convert to the proper units  
    semi_major_axis = semi_major_axis.to(unitL).value
    inclination = inclination.to(u.rad).value
    # compute the initial conditions 
    position_moon, velocity_moon = compute_moons_initial_conditions(semi_major_axis,eccentricity,inclination,G_val,const.Msun.value)
    # make the phase space 
    y0 = np.concatenate((position_moon, velocity_moon))
    # get the tidal tensor function
    unscaled_tidal_tensor_func = get_unscaled_kepler_tidal_tensor_func()
    # set the time span for integration 
    t_eval=np.linspace(t_span[0], t_span[1], t_eval_n_points)
    # solve the system of equations
    tidal_solution = integrate.solve_ivp(
        system_of_equations,
        t_span,
        y0,
        args=(G_val, Mearth, Dearth, omega, Msun, unscaled_tidal_tensor_func),
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval
    )
    rotating_two_body_solution = integrate.solve_ivp(
        two_body_rotating_frame_system_of_equations,
        t_span,
        y0,
        args=(G_val, Mearth, omega),
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval
    )
    return tidal_solution, rotating_two_body_solution