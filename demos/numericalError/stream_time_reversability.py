import numpy as np
import tstrippy
import datetime
import multiprocessing as mp
from astropy import units as u
from astropy import constants as const
from astropy import coordinates as coord
import os 
import h5py
import platform
import sys

outdirbase="/scratch2/sferrone/simulations/Stream/"
outdirbase="./simulations/"


# ANALYSIS FUNCTIONS 
def compute_stream_energy(MWparams,hostparams,hostkinematics,stream):
    """
    Compute the energy of the stream given the static galaxy potential and host kinematics.
    """
    # get the potential in the MW 
    _,_,_,phiMW = tstrippy.potentials.pouliasis2017pii(MWparams, stream[0], stream[1], stream[2])
    # get the total kinetic energy 
    T = 0.5 * (stream[3]**2 + stream[4]**2 + stream[5]**2)
    # get the distance to the host
    dx,dy,dz = stream[0] - hostkinematics[0], stream[1] - hostkinematics[1], stream[2] - hostkinematics[2]
    _,_,_,phiHost = tstrippy.potentials.plummer(hostparams, dx, dy, dz)
    return phiMW, phiHost, T

# EXPERIMENTS 
def experiment_stream_retrace(GCname, alphas, NP=int(512), integrationtime=1, comp_time_single_step_estimate=90e-8):

    """ 
    This experiment is designed to test the scaling of the computation time for integrating the most typical globular cluster in the Milky Way.
    Given a globular cluster, and an integration time, it will compute the stream and record how long it takes.
    This experiment will run in parallel for different numbers of particles and different time step scaling factors.
    The bottle neck of the program is basically the largest number of particles, and the largest time step scaling factor.
    The time step scaling factor is the fraction of the dynamical time that is used to compute
    the time step for the leapfrog integration.
    Doing it in parallel is a bit of a joke because the load balancing is horrible by design. 
    But that's ok, we're here to profile a code the speed. 
    """




    assert isinstance(targetGC, str), "targetGC must be a string"


    # print the estimated computation time for each simulation
    A = 4/3 
    B = 0.95

    # make alphas descending, do fast computations first
    alphas = np.sort(alphas)[::-1]

    # get the static galaxy, which doesn't change 
    MWparams = tstrippy.Parsers.pouliasis2017pii()
    staticgalaxy = ["pouliasis2017pii", MWparams ]
    currenttime = 0 
    clusterinitialkinematics = pick_GC_get_kinematics(GCname)
    G, M, rhm = load_hostparams(GCname)
    aplum = tstrippy.ergodic.convertHalfMassRadiusToPlummerRadius(rhm)
    tau = plummer_dynamical_time([G, M, aplum])
    hostparams= [G, M, aplum]
    attrs = {'GCname': GCname,"note": "experiment to get the conservation of energy in a stream retrace",}
    initialstream           =   tstrippy.ergodic.isotropicplummer(G,M,rhm,NP)
    
    for i in range(len(alphas)):
        alpha = alphas[i]
        integrationparameters   =   prepare_integration_arguments(currenttime=currenttime, integrationtime=integrationtime, tdyn=tau, alpha=alpha)
        NSTEPS = integrationparameters[-1]
        expected_time = comp_time_single_step_estimate * NP**(A) * NSTEPS ** B
        print(f"Expected computation time for {GCname} NPs={NP}, NSTEPS={NSTEPS}, alpha={alpha}: {expected_time:.2f} seconds")
        args=staticgalaxy, integrationparameters, clusterinitialkinematics, hostparams, initialstream, attrs
        generate_stream_retrace_stream_leapfrogtofinalpositions_and_save(args)



    



    return None

def experiment_stream_computation_time_scaling(targetGC, integrationtime, NPs, alphas, comp_time_single_step_estimate=5000e-9,ncpus = 6):

    """ 
    This experiment is designed to test the scaling of the computation time for integrating the most typical globular cluster in the Milky Way.
    Given a globular cluster, and an integration time, it will compute the stream and record how long it takes.
    This experiment will run in parallel for different numbers of particles and different time step scaling factors.
    The bottle neck of the program is basically the largest number of particles, and the largest time step scaling factor.
    The time step scaling factor is the fraction of the dynamical time that is used to compute
    the time step for the leapfrog integration.
    Doing it in parallel is a bit of a joke because the load balancing is horrible by design. 
    But that's ok, we're here to profile a code the speed. 
    """
    

    assert len(NPs) > 0, "NPs must not be empty"

    assert isinstance(targetGC, str), "targetGC must be a string"

    assert isinstance(NPs, (list, np.ndarray)), "NPs must be a list or numpy array"
    assert all(isinstance(n, (int, np.integer)) for n in NPs), "All NPs must be integers"

    # make the longer computations happen first 
    NPs = np.sort(NPs)[::-1]
    # make alphas ascending 
    alphas = np.sort(alphas)

    # get the static galaxy, which doesn't change 
    MWparams = tstrippy.Parsers.pouliasis2017pii()
    staticgalaxy = ['pouliasis2017pii', MWparams]


    # Extract the GC initial conditions 
    GCdata=tstrippy.Parsers.baumgardtMWGCs().data
    GCindex = np.where(GCdata['Cluster'] == targetGC)[0][0]
    Mass = GCdata['Mass'][GCindex].value
    rhm = GCdata['rh_m'][GCindex].value
    aplum = tstrippy.ergodic.convertHalfMassRadiusToPlummerRadius(rhm)
    G = MWparams[0]
    tau=plummer_dynamical_time([G,Mass,rhm])
    clusterinitialkinematics = pick_GC_get_kinematics(targetGC)    
    currenttime=0

    # PREPARE THE ARGUMENTS FOR EACH SIMULATION 
    
    # we want to make sure the same sphere is used for each simulation with the same particle number
    streamInitialKinematics = {}
    for i in range(len(NPs)):
        streamInitialKinematics[i] = np.array(tstrippy.ergodic.isotropicplummer(G, Mass, rhm, NPs[i]))
    
    integrationparameters = {}
    NSTEPS = []
    for i in range(len(alphas)):
        integrationparameters[i] = prepare_integration_arguments(
            currenttime=currenttime,
            integrationtime=integrationtime,
            tdyn=tau,
            alpha=alphas[i])
        NSTEPS.append(integrationparameters[i][-1])    

    attrs = {
    "GCname": targetGC,
    "Note": "An experiment testing the scaling of the computation for integrating the most typical GC"}
    # make into a 1D array for multiprocessing
    hostparams = [G, Mass, aplum]
    arguments = []
    for i in range(len(NPs)):
        for j in range(len(alphas)):
            args = (staticgalaxy, integrationparameters[j], clusterinitialkinematics, hostparams, streamInitialKinematics[i], attrs)
            arguments.append(args)
            expected_comptime = NSTEPS[j] * comp_time_single_step_estimate * NPs[i]
            print(f"Expected computation time for NPs={NPs[i]}, NSTEPS={NSTEPS[j]}: {expected_comptime:.2f} seconds")

    # make the pool of workers
    pool = mp.Pool(ncpus)
    # run the simulations in parallel
    starttime = datetime.datetime.now()
    pool.map(generate_stream_leapfrogtofinalpositions_and_save, arguments)
    endtime = datetime.datetime.now()
    print(f"All computations finished in {endtime - starttime} seconds")

    return None




##### DIFFERENT WAYS TO CREATE DATA  
def generate_stream_retrace_stream_leapfrogtofinalpositions_and_save(args):
    
    mystaticgalaxy, myintegrationparameters, myclusterinitialkinematics, myhostsparams, myinitialstream, attrs=args
    
    args = (mystaticgalaxy, myintegrationparameters, myclusterinitialkinematics, myhostsparams, myinitialstream)
    
    NP = len(myinitialstream[0])
    Nsteps = myintegrationparameters[-1]
    dirextension = "{:s}/numericalErrorExperiment/".format(mystaticgalaxy[0])
    fname= outdirbase + dirextension
    if not os.path.exists(fname):
        os.makedirs(fname, exist_ok=True)
    
    fname_base = "{:s}_stream_NSTEPS_{:d}_NP_{:d}_retrace.hdf5".format(attrs['GCname'], Nsteps, NP)
     
    fname = outdirbase + dirextension + fname_base
    
    if os.path.exists(fname):
        print(f"File {fname} already exists, skipping simulation.")
        return fname


    timestamps, timestamps_stream, hostorbit, streamfinal, tesc, comptimeorbit, comptimestream = generate_stream_leapfrogtofinalpositions(args)
    # now retrace the stream
    inithostperturber = [timestamps, *hostorbit, *myhostsparams]
    
    args = streamfinal, mystaticgalaxy, myintegrationparameters, inithostperturber
    
    stream_retrace, _, _, comptimeretrace = leapfrogtofinalpositions_stream_retrace(args)

    # save the results to a file

    # add more info about the processor to the attributes 
    attrs['processor'] = platform.processor()
    attrs['platform'] = platform.platform()
    attrs['python_version'] = platform.python_version()
    attrs['machine'] = platform.machine()
    attrs['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('timestamps', data=timestamps)
        f.create_dataset('timestamps_stream', data=timestamps_stream)
        f.create_dataset('hostorbit', data=hostorbit)
        f.create_dataset('streaminitial', data=myinitialstream)
        f.create_dataset('streamfinal', data=streamfinal)
        f.create_dataset('stream_retrace', data=stream_retrace)
        f.create_dataset('tesc', data=tesc)
        f.create_dataset('comptimeorbit', data=comptimeorbit)
        f.create_dataset('comptimestream', data=comptimestream)
        f.create_dataset('comptimeretrace', data=comptimeretrace)

        # save the attributes
        for key, value in attrs.items():
            f.attrs[key] = value

        # more attributes 
        f.attrs['potentialname'] = mystaticgalaxy[0]
        f.attrs['potentialparams'] = mystaticgalaxy[1]
        f.attrs['hostparams'] = myhostsparams
        f.attrs['integrationparameters'] = myintegrationparameters

    print(f"Saved results to {fname} in {comptimeretrace+comptimeorbit+comptimestream} seconds")

def generate_stream_leapfrogtofinalpositions_and_save(args):
    mystaticgalaxy, myintegrationparameters, myclusterinitialkinematics, myhostsparams, myinitialstream, attrs=args
    args = (mystaticgalaxy, myintegrationparameters, myclusterinitialkinematics, myhostsparams, myinitialstream)
    NP = len(myinitialstream[0])
    Nsteps = myintegrationparameters[-1]
    dirextension = "{:s}/numericalErrorExperiment/".format(mystaticgalaxy[0])
    fname= outdirbase + dirextension
    if not os.path.exists(fname):
        os.makedirs(fname, exist_ok=True)
    
    fname_base = "{:s}_stream_NSTEPS_{:d}_NP_{:d}_comp_time_experiment.hdf5".format(attrs['GCname'], Nsteps, NP)
     
    fname = outdirbase + dirextension + fname_base
    
    if os.path.exists(fname):
        print(f"File {fname} already exists, skipping simulation.")
        return fname


    timestamps, timestamps_stream, hostorbit, streamfinal, tesc, comptimeorbit, comptimestream = generate_stream_leapfrogtofinalpositions(args)

    # save the results to a file

    # add more info about the processor to the attributes 
    attrs['processor'] = platform.processor()
    attrs['platform'] = platform.platform()
    attrs['python_version'] = platform.python_version()
    attrs['machine'] = platform.machine()
    attrs['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('timestamps', data=timestamps)
        f.create_dataset('timestamps_stream', data=timestamps_stream)
        f.create_dataset('hostorbit', data=hostorbit)
        f.create_dataset('streaminitial', data=myinitialstream)
        f.create_dataset('streamfinal', data=streamfinal)
        f.create_dataset('tesc', data=tesc)
        f.create_dataset('comptimeorbit', data=comptimeorbit)
        f.create_dataset('comptimestream', data=comptimestream)

        # save the attributes
        for key, value in attrs.items():
            f.attrs[key] = value

        # more attributes 
        f.attrs['potentialname'] = mystaticgalaxy[0]
        f.attrs['potentialparams'] = mystaticgalaxy[1]
        f.attrs['hostparams'] = myhostsparams
        f.attrs['integrationparameters'] = myintegrationparameters

    print(f"Saved results to {fname}")


def generate_stream_leapfrogtofinalpositions(args):
    mystaticgalaxy, myintegrationparameters, myclusterinitialkinematics, myhostsparams, myinitialstream=args

    # DOUBLE THE TIMESTEPS FOR THE PERTURBER SO WE CAN GET THE POSITIONS AT THE INTERMEDIATE TIMESTEPS FOR UPDATING THE FORCE IN THE VELOCITIES:
    myintegrationparameters = list(myintegrationparameters)  # make it mutable
    NSTEPS_particles = myintegrationparameters[-1]
    NSTEPS_HOST = int(NSTEPS_particles * 2) 
    dt_PARTICLES = myintegrationparameters[1]  # time step for the particles
    dt_HOST = dt_PARTICLES / 2  # halve the time step for the host orbit
    myintegrationparameters[1] = dt_HOST
    myintegrationparameters[-1] = NSTEPS_HOST  # update the number of steps for
    hostorbit, timestamps, comptimeorbit = integrate_host_orbit_back([myclusterinitialkinematics, mystaticgalaxy, myintegrationparameters,])
    initialkinematics = myinitialstream + hostorbit[:,0][:,np.newaxis] # shift to the host's initial position 
    inithostperturber = [timestamps, *hostorbit, *myhostsparams ] # package the host orbit and parameters
    integrationparameters_stream = [timestamps[0], dt_PARTICLES, NSTEPS_particles]  # integration parameters for the stream
    
    streamfinal, tesc, timestamps_stream, comptimestream= leapfrogtofinalpositions_stream([initialkinematics, mystaticgalaxy, integrationparameters_stream, inithostperturber])
    return timestamps, timestamps_stream, hostorbit, streamfinal, tesc, comptimeorbit, comptimestream


def leapfrogtofinalpositions_stream_retrace(args):
    """ test the time reversibility of the leapfrog integration for the stream 
        Give the final position of the stream and the host orbit, it will retrace the stream and return the final positions and velocities.
        Let's see if we can get a normal plummer sphere back.
    INPUTS:
        initialkinematics: the initial kinematics of the stream, including the host orbit
        staticgalaxy: the static galaxy parameters
        integrationparameters: the integration parameters for the stream
        inithostperturber: the initial host orbit and parameters
    OUTPUTS:
        stream_retrace: the final positions and velocities of the stream after retracing
        tesc: the time of escape for the stream
        timestamps_retrace: the timestamps of the retraced stream
        comptime: the computation time for the retracing
    """
    initialkinematics, staticgalaxy, integrationparameters, inithostperturber = args
    # flip the sign of the velocities for the host orbit 
    inithostperturber = list(inithostperturber)  # make it mutable
    hostorbit = inithostperturber[1]
    hostorbit[3:6] = -hostorbit[3:6]  # flip the velocities
    inithostperturber[1] = hostorbit
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.inithostperturber(*inithostperturber)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setbackwardorbit()
    starttime= datetime.datetime.now()
    tstrippy.integrator.leapfrogtofinalpositions()
    endtime = datetime.datetime.now()
    xf, yf, zf = tstrippy.integrator.xf.copy(), tstrippy.integrator.yf.copy(), tstrippy.integrator.zf.copy()
    vxf, vyf, vzf = tstrippy.integrator.vxf.copy(), tstrippy.integrator.vyf.copy(), tstrippy.integrator.vzf.copy()
    tesc = tstrippy.integrator.tesc.copy()
    timestamps_retrace = tstrippy.integrator.timestamps.copy()
    tstrippy.integrator.deallocate()
    stream_retrace = np.array([xf, yf, zf, vxf, vyf, vzf])
    comptime = (endtime - starttime).total_seconds()
    return stream_retrace, tesc, timestamps_retrace, comptime


def leapfrogtofinalpositions_stream(args):
    initialkinematics, staticgalaxy, integrationparameters, inithostperturber = args
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setintegrationparameters(*integrationparameters)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.inithostperturber(*inithostperturber)
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    starttime= datetime.datetime.now()
    tstrippy.integrator.leapfrogtofinalpositions()
    endtime = datetime.datetime.now()
    xf, yf, zf = tstrippy.integrator.xf.copy(), tstrippy.integrator.yf.copy(), tstrippy.integrator.zf.copy()
    vxf, vyf, vzf = tstrippy.integrator.vxf.copy(), tstrippy.integrator.vyf.copy(), tstrippy.integrator.vzf.copy()
    tesc = tstrippy.integrator.tesc.copy()
    timestamps = tstrippy.integrator.timestamps.copy()
    tstrippy.integrator.deallocate()
    stream = np.array([xf, yf, zf, vxf, vyf, vzf])
    comptime = (endtime - starttime).total_seconds()
    return stream, tesc, timestamps, comptime


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



### HELPER FUNCTIONS
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


def load_hostparams(targetGC):
    # get the static galaxy, which doesn't change 
    MWparams = tstrippy.Parsers.pouliasis2017pii()
    # Extract the GC initial conditions 
    GCdata=tstrippy.Parsers.baumgardtMWGCs().data
    GCindex = np.where(GCdata['Cluster'] == targetGC)[0][0]
    Mass = GCdata['Mass'][GCindex].value
    rhm = GCdata['rh_m'][GCindex].value
    G = MWparams[0]
    return G, Mass, rhm

def pick_GC_get_kinematics(GCname):
    """
    Function to pick a globular cluster by name.
    """
    MWrefframe = tstrippy.Parsers.MWreferenceframe()
    GCdata=tstrippy.Parsers.baumgardtMWGCs().data
    GCindex = np.where(GCdata['Cluster'] == GCname)[0][0]
    x,y,z,vx,vy,vz  = load_globular_clusters_in_galactic_coordinates(MWrefframe)
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


if __name__ == "__main__":
    # Example usage
    # targetGC = 'NGC6760' # the weighted "most typical GC in the MW" by internal dynamical time and crossing time
    # targetGC = 'NGC6218'
    # targetGC = 'NGC6934'
    # targetGC = 'NGC6171'

    DO_experiment_stream_computation_time_scaling = False
    DO_experiment_stream_retrace = True 
    
    if DO_experiment_stream_computation_time_scaling:
        targetGC = sys.argv[1] 
        integrationtime = 1  # in dynamical time units
        NPs = np.logspace(1,4,4)  # number of particles for the stream
        NPs = np.array([int(np.floor(n)) for n in NPs],dtype=int)  # ensure they are integers
        alphas = np.logspace(0,-3,4)
        experiment_stream_computation_time_scaling(targetGC, integrationtime, NPs, alphas)

    if DO_experiment_stream_retrace:
        targetGC = sys.argv[1] 
        alphas = np.logspace(0,-2,4)
        NP = 512
        integrationtime = 1  # in dynamical time units
        comp_time_single_step_estimate = 90e-8  # estimated computation time for a single step
        experiment_stream_retrace(targetGC, alphas, NP, integrationtime, comp_time_single_step_estimate)