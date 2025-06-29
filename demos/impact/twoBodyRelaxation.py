import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp





def integrate_in_finite_box(NP, cutBmin=True, norbits=4, dtfactor=1/1000, x0 = np.array([-1/2, 0, 0]), v0 = np.array([1,0,0])):
    """Find the random walk of a particle in a gravitational field in a finite box
        This is normalized to the total mass of the system, which is m*NP
        and G =1 for simplicity, and the radius (width/2) of the system = 1. 
        to rescale, multiple v*sqrt(GM/a) where M = m*NP and a is the radius of the system.
        and multiple x*a
    """
    # Initialize positions and velocities
    x = np.random.uniform(-0.5, 0.5, NP)
    y = np.random.uniform(-0.5, 0.5, NP)
    z = np.random.uniform(-0.5, 0.5, NP)
    positions = np.array([x, y, z]).T
    # Initial velocities
    dt, nsteps = set_integration_parameters(norbits=norbits, dtfactor=dtfactor)
    speed = 1 # normalized to np.sqrt(GM/a) where M = m*NP
    # do the random walk computation
    v=np.zeros((nsteps+1, 3))
    x=np.zeros_like(v)
    v[0]= v0
    x[0] = x0
    for i in range(nsteps):
        dx= positions - x[i]
        speed = np.linalg.norm(v[i])
        quantity=np.dot(v[i],dx.T)/ speed**2
        bmin = 2/(NP*speed**2)
        # to be inside the plane perpendicular to the trajectory, the distance to the line must be less than dx
        condA = -quantity < dt
        condB = quantity < dt
        cond = np.logical_and(condA,  condB)
        # reproject for each impact parameter by replacing the particles on the line perpendicular to the trajectory
        # we want no drift in speed, just direction
        b = dx[cond] - np.outer(np.dot(dx[cond], v[i]), v[i])/ speed**2
        b_mag = np.linalg.norm(b, axis=1)
        # compute the minimum allowed impact parameter, which is the distance at which a binary would form
        if cutBmin:
            condC = b_mag > bmin
            b = b[condC]
            b_mag = b_mag[condC]
        if len(b) == 0:
            impulse = np.zeros((3,2))
        else:
            impulse = 2 / (NP*speed*b_mag**2) * b.T
        v[i+1] = v[i] + impulse.sum(axis=1)
        x[i+1] = x[i] + v[i+1] * dt
    return x, v, positions

def set_integration_parameters(norbits=4,dtfactor=1/1000):
    speed = 1 # normalized to np.sqrt(GM/a) where M = m*NP
    tdyn = 1 / speed
    dt = tdyn * dtfactor
    inttime = norbits * tdyn  
    nsteps = int( inttime / dt)
    return dt, nsteps

def deprojection_coordinate_system(vel):
    """ Create an arbitrary axes perpendicular to the trajectory """
    speed = np.linalg.norm(vel)
    unitV = vel / speed
    if abs(unitV[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    n1 = np.cross(unitV, temp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(unitV, n1)
    n2 /= np.linalg.norm(n2)
    return unitV, n1, n2

def deproject_plane_to_3D_space(u, v, position, vel, dt):
    """ u and v are the coordinates in the plane perpendicular to the trajectory, 
    deproject them into 3D space using the position and velocity of the particle."""
    unitV, n1, n2= deprojection_coordinate_system(vel)
    shift=np.random.uniform(-1/2,1/2)*dt*vel
    deprojected = position + u[:, np.newaxis] * n1 + v[:, np.newaxis] * n2 + shift
    return deprojected

def deproject_all_particles(us, ws, x, v, dt):
    """ Deproject all particles from the plane perpendicular to the trajectory into 3D space. """
    xdp = []
    ydp = []
    zdp = []
    for i in range(len(us)):
        deprojected = deproject_plane_to_3D_space(us[i], ws[i], x[i], v[i], dt).T
        nsamps = deprojected.shape[1]
        for j in range(nsamps):
            xdp.append(deprojected[0, j])
            ydp.append(deprojected[1, j])
            zdp.append(deprojected[2, j])
    return np.array(xdp), np.array(ydp), np.array(zdp)

def generate_uniform_disk(Ndisk):
    R = np.random.uniform(0, 1, Ndisk)**(1/2)
    theta = np.random.uniform(0, 2*np.pi, Ndisk)
    return R, theta
    

def change_in_velocity_from_all_particles_in_the_disk(NP, speed, u, w, cutBmin=True):
        """ Compute the forces on a particle in a disk perpendicular to the trajectory.
        NP is the number of particles in the disk, v is the velocity of the particle,
        b_mag is the impact parameter, and theta is the angle of the impact parameter.
        If cutBmin is True, the impact parameter is cut to be larger than the minimum
        impact parameter for binary formation.
        Returns the mean impulse and the u, w coordinates of the disk. """
        

        bmin = 2 / (NP * speed**2)
        b_mag = np.sqrt(u**2 + w**2)
        
        if cutBmin:
            cond = b_mag > bmin
            b_mag = b_mag[cond]
        
        if len(b_mag) == 0:
            # if there are no particles in the disk, skip this step
            mean_impulse = np.zeros(3)
            u=np.array([])
            w=np.array([])
        else:
            # now compute the forces on the particle in this plane 
            b_vec = np.array([np.zeros_like(u), u, w])
            # cut the impact parameter if it is too small
            impulse = 2 / (NP * speed * b_mag**2) * b_vec
            mean_impulse = impulse.mean(axis=1)

        return mean_impulse


def generate_disk_and_compute_impulse(NP, v, dt, cutBmin=True):
    """ Generate a disk of particles perpendicular to the trajectory and compute the impulse on the particle.
    NP is the number of particles in the disk, v is the velocity of the particle,
    dt is the time step, and cutBmin is a boolean indicating whether to cut the
    impact parameter to be larger than the minimum impact parameter for binary formation.
    Returns the mean impulse and the u, w coordinates of the disk. """
    speed = np.linalg.norm(v)
    if speed == 0:
        # if the speed is zero, there is no disk to generate
        return np.zeros(3), np.array([]), np.array([])
    
    numberDensity = NP / np.pi  # number of particles in the "cylinder" (N / (pi R^2 * v * T)) T = R/v
    
    Ndisk = np.random.poisson(numberDensity*speed*dt) # number of particles in the section of the disk
    
    b_mag, theta = generate_uniform_disk(Ndisk)    
    
    u, w = b_mag * np.cos(theta), b_mag * np.sin(theta) 
    
    impulse = change_in_velocity_from_all_particles_in_the_disk(NP, speed, u, w, cutBmin=cutBmin)

    return impulse, u, w



def experiment_always_in_the_center(NP, norbits=10, dtfactor=1/1000, x0=np.array([-1, 0, 0]), v0=np.array([1, 0, 0]),cutBmin=True):
    """ Run the experiment where particles are always within a certain distance of the trajectory. """
    dt, nsteps = set_integration_parameters(norbits=norbits, dtfactor=dtfactor)
    speed = 1 # starts @ 1 by normalization, but can be rescaled later
    numberDensity = NP / np.pi # number of particles in the "cylinder" (N / (pi R^2 * v * T)) T = R/v
    print("expectation in disk: ", numberDensity * speed * dt)
    bmin = 2/(NP*speed**2)  # Minimum impact parameter for binary formation
    x = np.zeros((nsteps+1, 3))
    v = np.zeros((nsteps+1, 3))
    x[0] = x0
    v[0] = v0
    us,ws = [],[]

    for i in range(nsteps):
        
        impulse, u, w = generate_disk_and_compute_impulse(NP, v[i], dt, cutBmin=cutBmin)

        unitV, n1, n2 = deprojection_coordinate_system(v[i])
        
        v[i+1,:] = v[i] + n1*impulse[1] + n2*impulse[2]
        
        x[i+1,:] = x[i] + v[i+1] * dt
        
        us.append(u)
        ws.append(w)
    return x, v, us, ws



def run_trial_experiment_always_in_the_center(args):
    NP, norbits, dtfactor, x0, v0, dt = args
    x, v, u, w = experiment_always_in_the_center(NP, norbits=norbits, dtfactor=dtfactor, x0=x0, v0=v0)
    xdp, ydp, zdp = deproject_all_particles(u, w, x, v, dt)
    return x, v, u, w, xdp, ydp, zdp