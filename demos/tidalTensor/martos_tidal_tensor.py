"""
This script will create some plots showing the 
stream coming from an cluster integrated in 
a martos halo potential.

"""

import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import thesis_rcparams
import tstrippy
from astropy import units as u
from astropy import constants as const
import datetime


def f(s,gamma):
    numerator = gamma -1 
    denominator = 1 + s**(gamma-1)
    return 2 - numerator/denominator

def mass_profile(s,gamma):
    numerator = s**gamma
    denominator = 1 + s**(gamma-1)
    return numerator/denominator

def compute_vcirc(haloparams, x0):
    G,Mo,rc,gamma,rcut = haloparams
    # get the initial veloicity
    Mr=mass_profile(x0/rc, gamma) * Mo 
    # compute the circular velocity
    v_circ = np.sqrt(G * Mr / x0) 
    return v_circ


def normalized_martos_tidal_tensor(r,gamma):
    """
    The position should be normalized to the scale radius
    r is the position vector of the satellite with respect to the system
    gamma is the mass envelope index
    """
    # Define the variables
    # assert isinstance(r,np.array), "a numpy array is expected"
    # assert isinstance(gamma, (int, float)), "a float or int is expected"
    # assert r.shape[0] == 3, "r must be a 3D vector"
    
    # get the norm of r and dr
    r_norm = np.linalg.norm(r)

    mass_enclosed = mass_profile(r_norm, gamma)
    f_s = f(r_norm, gamma)
    # make  the tensor
    tensor = np.zeros((3,3))
    # the diagonal elements
    tensor[0,0] = 1 -  f_s*(r[0]/r_norm)**2
    tensor[1,1] = 1 -  f_s*(r[1]/r_norm)**2
    tensor[2,2] = 1 -  f_s*(r[2]/r_norm)**2
    # the off diagonal elements
    tensor[0,1] = -f_s*(r[0]*r[1])/r_norm**2
    tensor[0,2] = -f_s*(r[0]*r[2])/r_norm**2
    tensor[1,0] = -f_s*(r[1]*r[0])/r_norm**2
    tensor[1,2] = -f_s*(r[1]*r[2])/r_norm**2
    tensor[2,0] = -f_s*(r[2]*r[0])/r_norm**2
    tensor[2,1] = -f_s*(r[2]*r[1])/r_norm**2
    # multiply by the mass enclosed
    tensor *= -(mass_enclosed/r_norm**3)
    return tensor

def magnitude_tidal_tensor(haloparams):
    """
    The position should be normalized to the scale radius
    r is the position vector of the satellite with respect to the system
    gamma is the mass envelope index
    """
    # unpact the halo params 
    G,Mo,rc,gamma,rcut = haloparams
    # magnitude 
    return G*Mo/rc**3

def get_dimensionless_deformed_ellipse(r0,dr,haloparams,dtfactor=500):
    """
    Everything is given in physical units 
    
    starting with a circle deform it by the tidal tensor
    The deformation is (1/2)*T_magn*dt^2
    dt is dynamicaltime/dtfactor, if you wanna scale the deformation, mess with dtfactor
    dr is the radius of the circle
    """
    # unpact the halo params 
    G,Mo,rc,gamma,rcut = haloparams
    # get the deformation tensor
    tensor = normalized_martos_tidal_tensor(r0/rc, gamma)
    # get the magnitude of the martos tensor
    scaling=magnitude_tidal_tensor(haloparams)
    tensor *= scaling
    # get the circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = dr*np.cos(theta)
    y = dr*np.sin(theta)
    z = np.zeros_like(x)
    xyz = np.vstack((x,y,z))
    # get the tidal force at each point
    dF_circ = np.dot(tensor, xyz)
    circle = np.zeros((3, len(theta)))
    circle[0] =  x
    circle[1] =  y
    circle[2] =  z
    # get the characteristic time to scale the force into distance 
    r0_mag = np.linalg.norm(r0)
    vcirc=compute_vcirc(haloparams,r0_mag)
    t_orbit = 2*np.pi*r0_mag/vcirc
    dtstep = t_orbit/dtfactor
    myellipse =np.zeros((3, len(theta)))
    myellipse[0] = x+np.sign(dF_circ[0])* (1/2)*np.sqrt(np.abs(dF_circ[0]*dtstep**2))
    myellipse[1] = y+np.sign(dF_circ[1])* (1/2)*np.sqrt(np.abs(dF_circ[1]*dtstep**2))
    myellipse[2] = z+np.sign(dF_circ[2])* (1/2)*np.sqrt(np.abs(dF_circ[2]*dtstep**2))
    return myellipse,circle




def make_stream(haloparams,orbitparams,isotropicplummer,dtfrac=1/50):
    """
    isotropicplummer = (G,M,r1/2,NP)
    orbitparams=x0,eccen,num_orbits
    
    """
    # parameters that shouldn't change 
    haloname = "allensantillianhalo"
    initial_time= 0.0
    MAXDATA = 10*u.Gbyte 

    # UNPACK THE PARAMETERS
    G,halomass,haloradius,gamma,rcut= haloparams
    G,Mhost,halfmassradius,NP = isotropicplummer
    x0,eccen,num_orbits=orbitparams

    # get the particle distribution
    xp,yp,zp,vxp,vyp,vzp = tstrippy.ergodic.isotropicplummer(*isotropicplummer)
    rplum = tstrippy.ergodic.convertHalfMassRadiusToPlummerRadius(halfmassradius)
    # COMPUTE THE TIME STEP 
    rps= np.sqrt(xp**2+yp**2+zp**2)
    vps= np.sqrt(vxp**2+vyp**2+vzp**2)
    tps = rps/vps
    chartime=np.median(tps)
    dt = chartime*dtfrac


    # get the initial conditions ofr the host orbit 
    
    v_circ= compute_vcirc(haloparams,x0)
    x0,y0,z0  = x0,0.0,0.0
    vx0,vy0,vz0 = 0.0, (1-eccen) * v_circ, 0.0
    initialkinematics = (x0, y0, z0, vx0, vy0, vz0)

    # set the integration parameters
    integration_time = 2*np.pi*num_orbits*(x0/v_circ)
    NSTEP = int(integration_time/dt)
    integrationparameters = (float(initial_time), dt,NSTEP)

    # compute the data size and print it for fun 
    datasize = NP*6*NSTEP*4  # 4 bytes per float
    datasize = datasize*u.byte  # Convert to byte units
    print(f"Data size is {datasize:.2f} bytes")
    if datasize > MAXDATA:
        print(f"\t which is larger than {MAXDATA:.2f}.")
        print("Please reduce the number of particles or the number of steps.")
        return None
    
    ### CLUSTER ORBIT 
    tstrippy.integrator.deallocate()
    staticgalaxy = (haloname, haloparams)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setbackwardorbit()
    xt,yt,zt,vxt,vyt,vzt = tstrippy.integrator.leapfrogintime(NSTEP,1)
    timesteps = tstrippy.integrator.timestamps.copy()
    # flip and change the sign of the velocities
    xt=xt[0,::-1]
    yt=yt[0,::-1]
    zt=zt[0,::-1]
    vxt=-vxt[0,::-1]
    vyt=-vyt[0,::-1]
    vzt=-vzt[0,::-1]
    timesteps=timesteps[::-1]
    tstrippy.integrator.deallocate()    

    #### INTEGRATE THE STREAM 
    initial_conditions=(xp+xt[0],yp+yt[0],zp+zt[0],vxp+vxt[0],vyp+vyt[0],vzp+vzt[0])
    # set the integration parameters
    integrationparameters = (timesteps[0],dt,NSTEP)
    # set the host perturber 
    inithostperturber=(timesteps,xt,yt,zt,vxt,vyt,vzt,Mhost,rplum)
    # do the integration 
    starttime=datetime.datetime.now()
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinitialkinematics(*initial_conditions)
    tstrippy.integrator.inithostperturber(*inithostperturber)
    xpt, ypt, zpt, vptx, vpty, vptz = tstrippy.integrator.leapfrogintime(NSTEP,NP)
    tstrippy.integrator.deallocate()
    endtime=datetime.datetime.now()
    print("Time taken to integrate the orbit: ", endtime-starttime)

    stream = (xpt, ypt, zpt, vptx, vpty, vptz)
    # get the host orbit
    hostorbit = (xt, yt, zt, vxt, vyt, vzt)
    
    return timesteps,  hostorbit, stream
