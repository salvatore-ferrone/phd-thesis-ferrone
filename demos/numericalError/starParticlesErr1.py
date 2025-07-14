import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tstrippy
import datetime
import multiprocessing as mp




def main(NP = 100, integrationtime = 10, alpha = 1/100):
    """
    Main function to run the star particle integration with error analysis.
    
    Parameters:
    NP : int
        Number of particles to simulate.
    integrationtime : float
        Total time for the integration in dynamical time units.
    alpha : float
        Time step scaling factor relative to the dynamical time.
    """
    
    initialkinematics, staticgalaxy, integrationparameters = prepare_integration_arguments(NP, integrationtime, alpha)
    
    # integrate the orbit
    stream_orbit, finalpositions = leapfrogintime(initialkinematics, staticgalaxy, integrationparameters)
    
    # calculate the relative energy error
    Erel = relative_energy_error_orbit(initialkinematics, staticgalaxy, stream_orbit)
    
    return stream_orbit, finalpositions, Erel



#### HELPER FUNCTIONS ####

def prepare_integration_arguments(NP = 100, integrationtime = 10, alpha = 1/100):

    # DEFINE THE PARAMETERS !!

    G,M,a = 1,1,1
    aplum = tstrippy.ergodic.convertHalfMassRadiusToPlummerRadius(a)
    tdyn = np.sqrt(a**3/(G*M))
    dt = alpha * tdyn
    NSTEP = int(integrationtime/dt)

    # sample the plummer sphere
    xp,yp,zp,vxp,vyp,vzp = tstrippy.ergodic.isotropicplummer(G,M,a,NP)
    
    initialkinematics = [xp,yp,zp,vxp,vyp,vzp]
    staticgalaxy = ['plummer', [G,M,aplum]]
    integrationparameters = [0, dt, NSTEP]
    return initialkinematics, staticgalaxy, integrationparameters




#### ANALYSIS FUNCTIONS ####

def relative_energy_error_orbit(initialkinematics, staticgalaxy, stream_orbit):
    """
    Calculate the relative energy error for each time step in the orbit.
    
    Parameters:
    initialkinematics : list
        Initial positions and velocities of the particles.
    staticgalaxy : list
        Parameters of the static galaxy potential.
    stream_orbit : np.ndarray
        The orbit data containing positions and velocities at each time step.
    
    Returns:
    np.ndarray
        Relative energy error at each time step.
    """
    
    NSTEP = stream_orbit.shape[2] - 1
    NP = len(initialkinematics[0])
    
    Erel = np.zeros((NSTEP + 1, NP))
    
    xp,yp,zp,vxp,vyp,vzp = initialkinematics
    _,_,_,phi0 = tstrippy.potentials.plummer(staticgalaxy[1],xp,yp,zp)
    T0 = 0.5 * (vxp**2 + vyp**2 + vzp**2)
    E0 = T0 + phi0

    for i in range(NSTEP + 1):
        xf, yf, zf, vxf, vyf, vzf = stream_orbit[:, :, i]
        _,_,_,phif = tstrippy.potentials.plummer(staticgalaxy[1],xf,yf,zf)
        Tf = 0.5 * (vxf**2 + vyf**2 + vzf**2)
        Ef = Tf + phif
        Erel[i] = np.abs((Ef - E0) / E0)
    
    return Erel

def realative_enery_error_final(initialkinematics, staticgalaxy, stream):
    xp,yp,zp,vxp,vyp,vzp = initialkinematics
    xf,yf,zf,vxf,vyf,vzf = stream
    _,_,_,phi0 = tstrippy.potentials.plummer(staticgalaxy[1],xp,yp,zp)
    _,_,_,phif = tstrippy.potentials.plummer(staticgalaxy[1],xf,yf,zf)
    T0 = 0.5 * (vxp**2 + vyp**2 + vzp**2)
    Tf = 0.5 * (vxf**2 + vyf**2 + vzf**2)
    E0 = T0 + phi0
    Ef = Tf + phif
    Erel = np.abs((Ef - E0) / E0)
    return Erel




##### INTEGRATION FUNCTIONS #####

def leapfrogintime(initialkinematics, staticgalaxy, integrationparameters):

    NSTEP = integrationparameters[2]
    NP = len(initialkinematics[0])
    
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    xt,yt,zt,vxt,vyt,vzt=tstrippy.integrator.leapfrogintime(NSTEP,NP)
    # also get the final positions 
    xf,yf,zf = tstrippy.integrator.xf.copy(), tstrippy.integrator.yf.copy(), tstrippy.integrator.zf.copy()
    vxf,vyf,vzf = tstrippy.integrator.vxf.copy(), tstrippy.integrator.vyf.copy(), tstrippy.integrator.vzf.copy()
    tstrippy.integrator.deallocate()
    stream_orbit = np.array([xt, yt, zt, vxt, vyt, vzt])
    stream_orbit = np.reshape(stream_orbit, (6, NP, NSTEP+1))
    
    return stream_orbit, np.array([xf, yf, zf, vxf, vyf, vzf])

def leapfrogtofinalpositions(initialkinematics, staticgalaxy, integrationparameters):
    """
    Main function to run the star particle integration with error analysis.
    
    Parameters:
    NP : int
        Number of particles to simulate.
    integrationtime : float
        Total time for the integration in dynamical time units.
    alpha : float
        Time step scaling factor relative to the dynamical time.
    """
    
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.leapfrogtofinalpositions()
    xf, yf, zf = tstrippy.integrator.xf.copy(), tstrippy.integrator.yf.copy(), tstrippy.integrator.zf.copy()
    vxf, vyf, vzf = tstrippy.integrator.vxf.copy(), tstrippy.integrator.vyf.copy(), tstrippy.integrator.vzf.copy()
    tstrippy.integrator.deallocate()
    stream = np.array([xf, yf, zf, vxf, vyf, vzf])
    return stream

    
    


