import numpy as np
import tstrippy
import datetime
import multiprocessing as mp
import numericalErrorFunctions as NEF
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import constants as const




def generate_stream(args):

    initialkinematics, staticgalaxy, integrationparameters, inithostperturber = args

    tstrippy.integrator.deallocate()
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinithostperturber(*inithostperturber)
    starttime= datetime.datetime.now()
    tstrippy.integrator.leapfrogtofinalpositions()
    endtime = datetime.datetime.now()
    xf, yf, zf = tstrippy.integrator.xf.copy(), tstrippy.integrator.yf.copy(), tstrippy.integrator.zf.copy()
    vxf, vyf, vzf = tstrippy.integrator.vxf.copy(), tstrippy.integrator.vyf.copy(), tstrippy.integrator.vzf.copy()
    timestamps = tstrippy.integrator.timestamps.copy()
    tstrippy.integrator.deallocate()
    stream = np.array([xf, yf, zf, vxf, vyf, vzf])
    comptime = (endtime - starttime).total_seconds()
    return stream, timestamps, comptime


def integrate_host_orbit_back(args):    
    """
    Integrate the host orbit using the leapfrog method.
    
    Parameters:
    initialkinematics : tuple
        Initial kinematics of the host orbit (xt, yt, zt, vxt, vyt, vzt).
    staticgalaxy : tuple
        Static galaxy parameters (potentialname, params).   
    integrationparameters : tuple
        Integration parameters (initialtime, dt, NSTEP).
    Returns:
    tuple
        Integrated positions and velocities (xt, yt, zt, vxt, vyt, vzt).
    """
    initialkinematics, staticgalaxy, integrationparameters = args
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setbackwardorbit()
    startime = datetime.datetime.now()
    xt, yt, zt, vxt, vyt, vzt = tstrippy.integrator.leapfrogintime(integrationparameters[2], 1)
    endtime = datetime.datetime.now()
    comptime = endtime - startime
    comptime = comptime.total_seconds()
    print("integration time: ", endtime - startime)
    xt, yt, zt = xt[0], yt[0], zt[0]
    vxt, vyt, vzt = vxt[0], vyt[0], vzt[0]
    # flip the sign of the velocities to get the forward orbit
    xt, yt, zt = xt[::-1], yt[::-1], zt[::-1]
    vxt, vyt, vzt = -vxt[::-1], -vyt[::-1], -vzt[::-1]
    # pack the orbit
    orbit = np.array([xt, yt, zt, vxt, vyt, vzt])
    timestamps = tstrippy.integrator.timestamps.copy()[::-1]
    tstrippy.integrator.deallocate()
    return orbit, timestamps, comptime

def adjust_dt_factor(alpha, tau, integrationtime):
    """
    Adjust the time step based on the dynamical time and the scaling factor.
    This ensures that the integration interval is evenly divisible by the time step.
    And that the factor to create the time step is less than or equal to the user provided alpha.
    This is to ensure that the time step is a fraction of the dynamical time.
    
    Parameters:    alpha : float
        Time step scaling factor relative to the dynamical time to make the time step
    tau : float
        Dynamical time of the system.
    integrationtime : float
        Total time for the integration in dynamical time units.
    Returns:
    float
        Adjusted time step.
    """

    dt_provisoire = alpha * tau
    assert (dt_provisoire) < integrationtime, "alpha * tau must be less than integrationtime"
    
    # get the number of dividions of the interval
    k = np.log2(integrationtime / alpha / tau ) + 1
    # round up to ensure that alpha_out is lower than alpha and evenly divides the integration time
    k = np.ceil(k)
    NSTEP = int(2**(k-1))
    alpha_adjusted = integrationtime / 2**(k-1) / tau
    return alpha_adjusted, NSTEP


def pick_GC_get_kinematics(GCname):
    """
    Function to pick a globular cluster by name.
    """
    MWrefframe = tstrippy.Parsers.MWreferenceframe()
    GCdata=tstrippy.Parsers.baumgardtMWGCs().data
    GCindex = np.where(GCdata['Cluster'] == GCname)[0][0]
    x,y,z,vx,vy,vz  = NEF.load_globular_clusters_in_galactic_coordinates(MWrefframe)
    xGC, yGC, zGC = x[GCindex], y[GCindex], z[GCindex]
    vxGC, vyGC, vzGC = vx[GCindex], vy[GCindex],vz[GCindex]
    initialkinematics = [[xGC], [yGC], [zGC], [vxGC], [vyGC], [vzGC]]
    return initialkinematics


def plummer_dynamical_time(plummer_params):
    """
    Prepare the integration arguments for the plummer sphere in the galactic potential 
    with the given parameters.

    Parameters:
    plummer_params : tuple
        Parameters of the plummer sphere (G, M, a).
    integrationtime : float
        Total time for the integration in the same units as the plummer sphere.
    alpha : float, optional
        Time step scaling factor relative to the dynamical time to make the time step (default is
    
    """
    G,M,a = plummer_params
    return np.sqrt(a**3/(G*M))
    

def prepare_integration_arguments(currenttime,integrationtime,tdyn,alpha):
    """
    Prepare the integration arguments for the plummer sphere in the galactic potential 
    with the given parameters.

    Parameters:
    currenttime : float
        Current time in the same units as the plummer sphere.
    integrationtime : float
        Total time for the integration in the same units as the plummer sphere.
    tdyn : float
        Dynamical time of the system.
    alpha : float, optional
        Time step scaling factor relative to the dynamical time to make the time step (default is 1/100).

    Returns:
    tuple
        Integration parameters (currenttime, dt, NSTEP).
    """
    alpha_adjusted, NSTEP = adjust_dt_factor(alpha, tdyn, integrationtime)
    dt = alpha_adjusted * tdyn

    return currenttime, dt, NSTEP

def load_globular_clusters_in_galactic_coordinates(MWrefframe):
    """Extract all initial conditions of the globular clusters and transform them the MW frame"""
    unitT, unitV, unitD, unitM, unitG, G = loadunits()
    GCdata  =   tstrippy.Parsers.baumgardtMWGCs().data
    skycoordinates=coord.SkyCoord(
        ra=GCdata['RA'],
        dec=GCdata['DEC'],
        distance=GCdata['Rsun'],
        pm_ra_cosdec=GCdata['mualpha'],
        pm_dec=GCdata['mu_delta'],
        radial_velocity=GCdata['RV'],)
    galacticcoordinates = skycoordinates.transform_to(MWrefframe)
    x,y,z=galacticcoordinates.cartesian.xyz.to(unitD).value
    vx,vy,vz=galacticcoordinates.velocity.d_xyz.to(unitV).value
    return x,y,z,vx,vy,vz


def loadunits():
    # Load the units
    unitbasis = tstrippy.Parsers.potential_parameters.unitbasis
    unitT=u.Unit(unitbasis['time'])
    unitV=u.Unit(unitbasis['velocity'])
    unitD=u.Unit(unitbasis['distance'])
    unitM=u.Unit(unitbasis['mass'])
    unitG=u.Unit(unitbasis['G'])
    G = const.G.to(unitG).value
    return unitT, unitV, unitD, unitM, unitG, G
