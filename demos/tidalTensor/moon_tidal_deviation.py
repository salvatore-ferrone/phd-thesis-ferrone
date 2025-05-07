import sympy as sp 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy import integrate
import thesis_rcparams
import os
import multiprocessing as mp
import datetime

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
    G_val=const.G.to(unitG).value
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

def system_of_equations(t, y, G_val, Mearth, Dearth, omega, M_sun, unscaled_tidal_tensor_func):
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
    tidal_tensor = -(G_val*M_sun/r_earth**3)*unscaled_tidal_tensor_func(r_earth, position_earth[0], position_earth[1], position_earth[2])
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

def sun_and_earth(unitL, unitT, unitM,):
    # get the earth and sun 
    Dearth = 1*unitL
    omega = 2*np.pi/(1*unitT) 
    M_sun = const.M_sun.to(unitM).value
    Mearth = const.M_earth.to(unitM).value
    Dearth = Dearth.to(unitL).value
    omega = omega.value
    return M_sun,  Mearth, Dearth, omega

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
    M_sun,  Mearth, Dearth, omega= sun_and_earth(unitL, unitT, unitM)
    # get the moons orbital elements 
    semi_major_axis, eccentricity, inclination = get_moons_orbital_elements()
    # convert to the proper units  
    semi_major_axis = semi_major_axis.to(unitL).value
    inclination = inclination.to(u.rad).value
    # compute the initial conditions 
    position_moon, velocity_moon = compute_moons_initial_conditions(semi_major_axis,eccentricity,inclination,G_val,Mearth)
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
        args=(G_val, Mearth, Dearth, omega, M_sun, unscaled_tidal_tensor_func),
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

def obtain_earths_orbit(t_eval, Dearth, omega):
    """
    get the position of the earth as a function of time 
    """
    # get the position of the earth as a function of time 
    x_earth = Dearth * np.cos(omega * t_eval)
    y_earth = Dearth * np.sin(omega * t_eval)
    z_earth = np.zeros_like(t_eval)
    earthsOrbit=np.array([x_earth, y_earth, z_earth])
    return earthsOrbit

def set_plot_properties(semi_major_axis,):
    AXIS1 = {
        "xticks": np.arange(-1.5*semi_major_axis,1.5*semi_major_axis,semi_major_axis/2),
        "yticks": np.arange(-1.5*semi_major_axis,1.5*semi_major_axis,semi_major_axis/2),
        "xlim": (-1.5*semi_major_axis, 1.5*semi_major_axis),
        "ylim": (-1.5*semi_major_axis, 1.5*semi_major_axis),
        "xlabel": r"$x$ (Earth-Moon distance)",
        "ylabel": r"$y$ (Earth-Moon distance)",
        "title": "Non-Inertial Frame with Tidal Field",
    }
    AXIS1['xticklabels'] = [f'{x:.1f}' for x in AXIS1['xticks']/semi_major_axis]
    AXIS1['yticklabels'] = [f'{y:.1f}' for y in AXIS1['yticks']/semi_major_axis]
    AXIS2 = {
        "xticks": np.arange(-1.1,1.1,0.02),
        "yticks": np.arange(-1.1,1.1,0.02),
        "xlabel": r"$x$ [au]",
        "ylabel": r"$y$ [au]",
        "title":"Inertial frame",
    }

    PLOT_TIDAL = {"color": "tab:green", "label": None}
    PLOT_NONTIDAL = {"color": "tab:orange", "label": None}
    SCAT_TIDAL = {"color": "tab:green", "s": 10, "label": "Tides","zorder":10}
    SCAT_NONTIDAL = {"color": "tab:orange", "s": 10, "label": "No tides","zorder":10}
    QUIV_TIDAL = {"scale": 35, "label": None}
    QUIV_CENTRIFUGAL = {"scale": 35, "label": "Centrifugal Field"} 
    return AXIS1, AXIS2, PLOT_TIDAL, PLOT_NONTIDAL, SCAT_TIDAL, SCAT_NONTIDAL, QUIV_TIDAL, QUIV_CENTRIFUGAL

def do_plot(t_eval, down_index, sim_index, tidal_orbit, rotating_orbit, earthsOrbit, quiver, PROPERTIES):


    # ficticious start date
    ficticious_start_date = datetime.datetime(2022, 10, 1)
    years_forward = t_eval[sim_index]
    ficticious_date = ficticious_start_date + datetime.timedelta(days=365*years_forward)
    # get string of just the month and year as YYYY MM 
    ficticious_date_str = ficticious_date.strftime("%d %b %Y ")
    # unpack 
    X, Y, tidal_force, tidal_force_magnitude,  colors_tidal= quiver
    AXIS1, AXIS2, PLOT_TIDAL, PLOT_NONTIDAL, SCAT_TIDAL, SCAT_NONTIDAL, QUIV_TIDAL, QUIV_CENTRIFUGAL= PROPERTIES

    # create the inertial orbit 
    inertial_tidal_solution = np.zeros_like(tidal_orbit)
    inertial_rotating_solution = np.zeros_like(rotating_orbit)
    inertial_tidal_solution = tidal_orbit + earthsOrbit
    inertial_rotating_solution = rotating_orbit + earthsOrbit
    
    # Create figure with fixed dimensions
    fig = plt.figure(figsize=(12, 6))
    
    # Create axes with fixed positions - these values leave room for all labels
    # Format: [left, bottom, width, height] - all values are in figure fraction (0-1)
    ax0 = fig.add_axes([0.10, 0.12, 0.38, 0.75])  # Left plot
    ax1 = fig.add_axes([0.55, 0.12, 0.38, 0.75])  # Right plot
    ax = [ax0, ax1]  # Create a list to maintain compatibility with your code
    

    ax[1].plot(inertial_tidal_solution[0,down_index:sim_index+1], inertial_tidal_solution[1,down_index:sim_index+1], **PLOT_TIDAL)
    ax[1].scatter(inertial_tidal_solution[0,sim_index], inertial_tidal_solution[1,sim_index], **SCAT_TIDAL)
    ax[1].plot(inertial_rotating_solution[0,down_index:sim_index+1], inertial_rotating_solution[1,down_index:sim_index+1], **PLOT_NONTIDAL)
    ax[1].scatter(inertial_rotating_solution[0,sim_index], inertial_rotating_solution[1,sim_index], **SCAT_NONTIDAL)
    ax[1].plot(earthsOrbit[0,down_index:sim_index+1], earthsOrbit[1,down_index:sim_index+1],color='tab:blue', )
    ax[1].scatter(earthsOrbit[0,sim_index], earthsOrbit[1,sim_index],  color='tab:blue', label='Earth')
    # set the xlim ylim for axis2
    semi_major_axis=AXIS1["xlim"][1] - AXIS1["xlim"][0]
    AXIS2["xlim"]= (earthsOrbit[0,sim_index]-5*semi_major_axis, earthsOrbit[0,sim_index]+5*semi_major_axis)
    AXIS2["ylim"]= (earthsOrbit[1,sim_index]-5*semi_major_axis, earthsOrbit[1,sim_index]+5*semi_major_axis)
    ax[1].legend()
    # turn off the label
    # new_SCAT_TIDAL = SCAT_TIDAL.copy()
    # new_SCAT_NON_TIDAL = SCAT_NONTIDAL.copy()
    # new_SCAT_TIDAL['label'] = None
    # new_SCAT_NON_TIDAL['label'] = None
    ax[0].quiver(X.flatten(), Y.flatten(), tidal_force[0]/tidal_force_magnitude, tidal_force[1]/tidal_force_magnitude,color=colors_tidal, **QUIV_TIDAL)
    ax[0].scatter(0,0, color='tab:blue', label=None)
    ax[0].plot(tidal_orbit[0,down_index:sim_index+1], tidal_orbit[1,down_index:sim_index+1], **PLOT_TIDAL)
    ax[0].scatter(tidal_orbit[0,sim_index], tidal_orbit[1,sim_index], **SCAT_TIDAL)
    ax[0].plot(rotating_orbit[0,down_index:sim_index+1], rotating_orbit[1,down_index:sim_index+1], **PLOT_NONTIDAL)
    ax[0].scatter(rotating_orbit[0,sim_index], rotating_orbit[1,sim_index], **SCAT_NONTIDAL)
    
    # add the current time
    ax[0].text(0.025, 0.95, f"{ficticious_date_str}", transform=ax[0].transAxes, ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.75))

    ax[1].set(**AXIS2);
    ax[0].set(**AXIS1);
    # so the graph doesn't wobble when the x-ticks go to the right side
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].tick_params(axis='y', pad=10)
    ylabels = [item.get_text() for item in ax[1].get_yticklabels()]
    new_ylabels = []
    # add white space for positive numbers 
    for label in ylabels:
        # If it doesn't start with '-', add a space to match width of negative numbers
        if not label.startswith('-') and label.strip():
            new_ylabels.append(' ' + label)
        else:
            new_ylabels.append(label)
    ax[1].set_yticklabels(new_ylabels)
    return fig, ax

def plot_and_save(t_eval, down_index, sim_index, tidal_orbit, rotating_orbit, earthsOrbit, quiver, properties, output_path):
    """Wrapper function that calls do_plot and saves the result."""
    figtitle="Tidal Induced Orbital Drift of the Moon"
    fig, ax = do_plot(t_eval, down_index, sim_index, tidal_orbit, rotating_orbit, earthsOrbit, quiver, properties)
    fig.suptitle(figtitle, fontsize=16)

    dpi = 300
    # Start with desired pixel dimensions (must be even)
    desired_width_pixels = 3600  # Even number
    desired_height_pixels = 1800  # Even number

    # Convert to inches for matplotlib
    width_inches = desired_width_pixels / dpi
    height_inches = desired_height_pixels / dpi
    
    # Update figure size if needed
    fig.set_size_inches(width_inches, height_inches)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)  # Important to avoid memory leaks
    if sim_index % 100 == 0:
        print(f"saved {output_path}")
    
def coordinate_frame_to_simulation_index(frame_index, t_eval, FRAMES_PER_UNIT_TIME):
    SIM_TIME = frame_index / FRAMES_PER_UNIT_TIME
    sim_index = np.argmin(np.abs(t_eval - SIM_TIME))
    return sim_index

def main():
    # simulation params 
    outdir = "../frames/"
    os.makedirs(outdir, exist_ok=True)
    t_span = (0, 4)
    t_eval_n_points = 5000
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_n_points)

    #### Animation params  set some animation parametes
    FPS = 30
    SECONDS_PER_UNIT_TIME = 24 # which is that six seconds go by for each year 
    DURATION_IN_UNIT_TIME = t_eval[-1]
    TOTAL_SIM_INDEXES = len(t_eval)
    DURATION_IN_SECONDS = DURATION_IN_UNIT_TIME*SECONDS_PER_UNIT_TIME
    TOTAL_FRAMES = int(DURATION_IN_SECONDS * FPS)
    FRAMES_PER_UNIT_TIME = int(FPS * SECONDS_PER_UNIT_TIME)
    print("TOTAL_FRAMES", TOTAL_FRAMES)
    print("DURATION_IN_SECONDS", DURATION_IN_SECONDS)
    #### plotting params
    # width of the orbit back 
    orbit_index_width = 45
    # number of vectors to use in the vector field 
    nvec = 20    

    ### THE STUFF THAT DOESN'T CHANGE FRAME TO FRAME 
    # the units, sun earth and moon parameters 
    unitL, unitT, unitV, unitM, unitG, G_val = set_units()
    M_sun,  Mearth, Dearth, omega= sun_and_earth(unitL, unitT, unitM)
    semi_major_axis, _, _= get_moons_orbital_elements()
    semi_major_axis=semi_major_axis.to(unitL).value
    # solve the equations of motion
    tidal_solution, rotating_two_body_solution = solve_moons_equation_of_motion(t_span=t_span, t_eval_n_points=t_eval_n_points)
    # translate the positions to the earth's position 
    earthsOrbit = obtain_earths_orbit(t_eval, Dearth, omega)
    # Load the tidal tensor
    unscaled_tidal_tensor_func = get_unscaled_kepler_tidal_tensor_func()
    # make a vector field about the earth 
    xs= np.linspace(-1.5*semi_major_axis, 1.5*semi_major_axis, nvec)
    ys= np.linspace(-1.5*semi_major_axis, 1.5*semi_major_axis, nvec)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    # put the positions in a grid 
    positions = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    omega_vec = np.array([0, 0, omega])    

    # load in the plot properties 
    AXIS1, AXIS2, PLOT_TIDAL, PLOT_NONTIDAL, SCAT_TIDAL, SCAT_NONTIDAL, QUIV_TIDAL, QUIV_CENTRIFUGAL = set_plot_properties(semi_major_axis)

    # CREATE THE TIDAL TENSOR ONCE JUST TO GET THE PLOT PROPERTIES 
    frame_index = 0
    sim_index = coordinate_frame_to_simulation_index(frame_index, t_eval, FRAMES_PER_UNIT_TIME)
    tidal_tensor=(G_val*M_sun/Dearth**3)*unscaled_tidal_tensor_func(np.linalg.norm(earthsOrbit[:,sim_index]), earthsOrbit[0,sim_index], earthsOrbit[1,sim_index], earthsOrbit[2,sim_index])
    # get the tidal tensor at each position
    tidal_force = tidal_tensor.dot(positions.T)
    # evaluate the centrifugal force
    centrifugalforce= -np.cross(omega_vec, np.cross(omega_vec, positions)).T
    # normalize the vectors
    tidal_force_magnitude = np.linalg.norm(tidal_force, axis=0)
    centrifugalforce_magnitude = np.linalg.norm(centrifugalforce, axis=0)
    # set the color map
    vmax=np.max([centrifugalforce_magnitude.max(), tidal_force_magnitude.max()])
    vmin=np.min([centrifugalforce_magnitude.min(), tidal_force_magnitude.min()])
    cmap_tidal = mpl.colormaps.get_cmap('rainbow_r')
    cmap_centrifugal = mpl.colormaps.get_cmap('gray')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    colors_tidal = cmap_tidal(norm(tidal_force_magnitude))
    colors_centrifugal = cmap_centrifugal(norm(centrifugalforce_magnitude))


    ## BEGIN MULTIPROCESSING OVER THE FRAMES 
    frame_args = [ ]
    for frame_index in range(0,TOTAL_FRAMES):
        # compute down index to up index
        sim_index= coordinate_frame_to_simulation_index(frame_index, t_eval, FRAMES_PER_UNIT_TIME)
        down_index = sim_index - orbit_index_width
        up_index = sim_index 
        if down_index < 0:
            down_index = 0
        if sim_index >=TOTAL_SIM_INDEXES:
            up_index = TOTAL_SIM_INDEXES - 1    

        # create the tidal tensor
        tidal_tensor=(G_val*M_sun/Dearth**3)*unscaled_tidal_tensor_func(np.linalg.norm(earthsOrbit[:,sim_index]), earthsOrbit[0,sim_index], earthsOrbit[1,sim_index], earthsOrbit[2,sim_index])
        # get the tidal tensor at each position
        tidal_force = tidal_tensor.dot(positions.T)
        # evaluate the centrifugal force
        centrifugalforce= -np.cross(omega_vec, np.cross(omega_vec, positions)).T
        # normalize the vectors
        tidal_force_magnitude = np.linalg.norm(tidal_force, axis=0)
        centrifugalforce_magnitude = np.linalg.norm(centrifugalforce, axis=0)
        # update the frame for the earth since it's moving 
        # pack up the arugments 
        quiver = (X, Y, tidal_force, tidal_force_magnitude, colors_tidal)
        PROPERTIES = (AXIS1, AXIS2, PLOT_TIDAL, PLOT_NONTIDAL, SCAT_TIDAL, SCAT_NONTIDAL, QUIV_TIDAL, QUIV_CENTRIFUGAL)
        # pack up the arguments
        tidal_orbit = tidal_solution.y[0:3]
        rotating_orbit = rotating_two_body_solution.y[:3]
        # pack up the arguments
        frame_args.append((t_eval, down_index, sim_index, tidal_orbit, rotating_orbit, earthsOrbit, quiver, PROPERTIES, outdir+f"frame_{frame_index:04d}.png"))

    print("ARGUMENTS LOADED ")
    # use multiprocessing to plot the frames
    starttime= datetime.datetime.now()
    print("STARTING MULTIPROCESSING AT ", starttime)

    ncpu = mp.cpu_count()
    with mp.Pool(processes=ncpu) as pool:  # Use 4 processes
        pool.starmap(plot_and_save, frame_args)

        # plot_and_save(
        #     down_index, sim_index, tidal_orbit, rotating_orbit, earthsOrbit, quiver, PROPERTIES,
        #     output_path=outdir+f"frame_{frame_index:04d}.png"
        # )
    endtime= datetime.datetime.now()
    print("FINISHED MULTIPROCESSING AT ", endtime)
    print("TIME TAKEN FOR MULTIPROCESSING ", endtime-starttime)
    print("time per frame", (endtime-starttime)/len(frame_args))
if __name__ == "__main__":
    main()