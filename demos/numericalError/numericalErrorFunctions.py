from astropy import units as u
from astropy import constants as const
from astropy import coordinates as coord
import tstrippy
import numpy as np 

def get_dr_dv_rmean_vmean(backwardOrbit,forwardOrbit):
    """
    Calculate the difference in position and velocity between the backward and forward integration
    """
    dx=backwardOrbit[1]-forwardOrbit[1]
    dy=backwardOrbit[2]-forwardOrbit[2]
    dz=backwardOrbit[3]-forwardOrbit[3]
    dr=np.sqrt(dx**2+dy**2+dz**2)
    dvx=backwardOrbit[4]-forwardOrbit[4]
    dvy=backwardOrbit[5]-forwardOrbit[5]
    dvz=backwardOrbit[6]-forwardOrbit[6]
    dv=np.sqrt(dvx**2+dvy**2+dvz**2)
    xmean=(backwardOrbit[1]+forwardOrbit[1])/2
    ymean=(backwardOrbit[2]+forwardOrbit[2])/2
    zmean=(backwardOrbit[3]+forwardOrbit[3])/2
    vxmean=(backwardOrbit[4]+forwardOrbit[4])/2
    vymean=(backwardOrbit[5]+forwardOrbit[5])/2
    vzmean=(backwardOrbit[6]+forwardOrbit[6])/2
    rmean=np.sqrt(xmean**2+ymean**2+zmean**2)
    vmean=np.sqrt(vxmean**2+vymean**2+vzmean**2)
    return dr,dv,rmean,vmean


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


def get_energy_error(backwardOrbit,MWparams):
    """
    Calculate the energy error of the backward orbit for each globular cluster.
    """
    # calculate the potential and kinetic energy of the backward orbit
    phi = np.zeros_like(backwardOrbit[1])
    T = np.zeros_like(backwardOrbit[1])
    nGC = backwardOrbit[1].shape[0]
    for cluster_index in range(nGC):
        _,_,_,phi_=tstrippy.potentials.pouliasis2017pii(MWparams,backwardOrbit[1][cluster_index],backwardOrbit[2][cluster_index],backwardOrbit[3][cluster_index])
        phi[cluster_index] = phi_
        T[cluster_index] = (backwardOrbit[3][cluster_index]**2 + backwardOrbit[4][cluster_index]**2 + backwardOrbit[5][cluster_index]**2)/2
    
    E    = phi + T
    E0   = E[:, -1][:, np.newaxis]  # Take the first column as the reference energy
    dE   = E - E0
    errE = np.sqrt(dE**2) / E0
    return E0, errE


def vanilla_clusters(integrationtime,timestep,staticgalaxy,initialkinematics):
    """
    do the backward and forward integration of the vanilla clusters
    """
    assert isinstance(integrationtime,u.Quantity)
    assert isinstance(timestep,u.Quantity)
    unitT, unitV, unitD, unitM, unitG, G = loadunits()
    Ntimestep=int(integrationtime.value/timestep.value)
    dt=timestep.to(unitT)
    currenttime=0*unitT
    integrationparameters=[currenttime.value,dt.value,Ntimestep]
    nObj = initialkinematics[0].shape[0]

    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setbackwardorbit()
    xBackward,yBackward,zBackward,vxBackward,vyBackward,vzBackward=\
        tstrippy.integrator.leapfrogintime(Ntimestep,nObj)
    tBackward=tstrippy.integrator.timestamps.copy()
    tstrippy.integrator.deallocate()

    #### Now compute the orbit forward
    #### IT'S VERY IMPORTANT TO USE tBackward[-1] AS THE CURRENT TIME FOR THE FORWARD INTEGRATION
    #### BEFORE I USED -integrationtime, WHICH CAN BE DIFFERENT BY NSTEP * 1e-16
    #### I.e. A DRIFT IN TIME DUE TO NUMERICAL ERROR, WHICH CAN BECOME SIGNIFICANT FOR INTEGRATING WITH THE BAR
    currenttime=tBackward[-1]*unitT

    integrationparameters=[currenttime.value,dt.value,Ntimestep]
    x0,y0,z0=xBackward[:,-1],yBackward[:,-1],zBackward[:,-1]
    vx0,vy0,vz0 = -vxBackward[:,-1],-vyBackward[:,-1],-vzBackward[:,-1]
    initialkinematics=[x0,y0,z0,vx0,vy0,vz0]
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    xForward,yForward,zForward,vxForward,vyForward,vzForward=\
        tstrippy.integrator.leapfrogintime(Ntimestep,nObj)
    tForward=tstrippy.integrator.timestamps.copy()
    tstrippy.integrator.deallocate()

    # flip the backorbits such that the point in the past,
    # which should be the common starting point,
    # is the first point for both the forward and backward orbits
    tBackward=tBackward[::-1]
    xBackward,yBackward,zBackward=xBackward[:,::-1],yBackward[:,::-1],zBackward[:,::-1]
    vxBackward,vyBackward,vzBackward=-vxBackward[:,::-1],-vyBackward[:,::-1],-vzBackward[:,::-1]
    backwardOrbit  = [tBackward,xBackward,yBackward,zBackward,vxBackward,vyBackward,vzBackward]
    forwardOrbit   = [tForward,xForward,yForward,zForward,vxForward,vyForward,vzForward]
    return backwardOrbit,forwardOrbit

def experiment_vanilla_clusters_single_timestep(args):
    """
    intended for multiprocessing, run the vanilla clusters experiment for a single timestep
    and return the results.
    The arguments are passed as a tuple:
    (timestep, integrationtime, staticgalaxy, initialkinematics, MWparams)
    where:
    - timestep: the time step for the integration (astropy Quantity)
    - integrationtime: the total time for the integration (astropy Quantity)
    - staticgalaxy: a tuple containing the potential name and parameters
    - initialkinematics: a tuple containing the initial positions and velocities of the globular clusters
    - MWparams: the parameters of the Milky Way potential (used for energy error calculation)
    """
    # Unpack all arguments
    timestep, integrationtime, staticgalaxy, initialkinematics, MWparams = args

    import datetime  # Needed for multiprocessing on some systems
    starttime = datetime.datetime.now()
    backwardOrbit, forwardOrbit = vanilla_clusters(
        integrationtime, timestep, staticgalaxy, initialkinematics
    )
    endtime = datetime.datetime.now()
    computation_time = endtime - starttime

    dr, dv, rmean, vmean = get_dr_dv_rmean_vmean(backwardOrbit, forwardOrbit)
    E0, errE = get_energy_error(backwardOrbit, MWparams)
    idx = np.argsort(E0)
    dr, dv, rmean, vmean = dr[idx], dv[idx], rmean[idx], vmean[idx]
    E0 = E0[idx]
    errE = errE[idx]
    relative_R = dr / rmean
    relative_V = dv / vmean
    t = backwardOrbit[0]
    print(f"Finished integration for timestep {timestep} in {computation_time}")
    return computation_time, relative_R, relative_V, t, errE, E0
