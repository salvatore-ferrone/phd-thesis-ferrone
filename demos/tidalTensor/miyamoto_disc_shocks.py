import tstrippy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import pouliasis2017pii as p2017pii
import multiprocessing as mp
mpl.rcParams['text.usetex']=True

pagewidth,pageheight=11.75,8.5
potential = tstrippy.potentials.miyamotonagai

def get_tidal_eigenvalues_along_orbit(params,tensor_func,orbit,):
    """How to scale the transition from position to force when deforming the sphere.
    I want the strongest compression to flatten the sphere, but not invert it.
    """
    # unpack the orbit
    xt,yt,zt,vxt,vyt,vzt = orbit
    eigenvalues = np.zeros((len(xt),3))
    # evaluate the tensor at each point
    for i in range(len(xt)):
        tensor = tensor_func(params,xt[i],yt[i],zt[i])
        # get the eigenvalues
        eig = np.linalg.eigvals(tensor)
        # sort for stretching and then compressing
        eigenvalues[i] = np.sort(eig)[::-1]
    return eigenvalues


def make_and_save_figure(apocenter, inclination, pseudo_e, bp, dtfactor, Ndyntime, potentialname, initialtime, dx):

    fig, axis,axis2 = compute_orbit_and_shocks_and_make_figure(
        apocenter=apocenter,
        inclination=inclination,
        pseudo_e=pseudo_e,
        bp=bp,
        dtfactor=dtfactor,
        Ndyntime=Ndyntime,
        potentialname=potentialname,
        initialtime=initialtime,
        dx=dx
    )
    outfname = "../tidalTensorImages/miyamoto_disc_shocks_ab_rp_e_i_{:0.2f}_{:0.1f}_{:0.2f}_{:0.1f}.png".format(bp, apocenter, pseudo_e, inclination)

    fig.subplots_adjust(bottom=0.15)  # Increase if needed

    fig.savefig(outfname, dpi=300)
    print(f"Saved figure to {outfname}")
    plt.close(fig)


def compute_orbit_and_shocks_and_make_figure(
    apocenter       =   4,
    inclination     =   45,
    pseudo_e        =   0.5,  # pseudo eccentricity, always 0<e<1
    bp              =   1/10,   # ratio of b/a, scale height to scale radius
    dtfactor        =   1/1000, # time step factor, dt = t_dyn/dtfactor 
    Ndyntime        =   4,   # about how many orbits to do
    potentialname   =   'miyamotonagai',
    initialtime     =   0.0,
    dx              =   1/(50), # the size of the sphere to be deformed     
):


    # G,M,a,b
    params = [1.0,1.0,1.0,bp]


    orbitinfostring = r"a/b, r$_a$, e, i = {0:.2f}, {1:.1f}, {2:.2f}, {3:.1f}$^\circ$".format(bp, apocenter, pseudo_e, inclination)


    # pack up the initial conditions 
    initialkinematics = p2017pii.get_initial_conditions(potential,params, apocenter, inclination, pseudo_e)
    pos,vo = initialkinematics[0:3], initialkinematics[3:6]
    dt, Nsteps, t_dyn = p2017pii.get_time_info(pos,vo,dtfactor,Ndyntime)
    # compute the orbit 
    staticgalaxy            =   potentialname,params
    integrationparams       =   (initialtime, dt, Nsteps)
    print("integrationparams: ", integrationparams)
    print("staticgalaxy: ", staticgalaxy)
    print("initialkinematics: ", initialkinematics)



    # compute the orbit 
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setintegrationparameters(*integrationparams)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    xt,yt,zt,vxt,vyt,vzt    =   tstrippy.integrator.leapfrogintime(Nsteps,1)
    xt,yt,zt                =   xt[0],yt[0],zt[0]
    vxt,vyt,vzt             =   vxt[0],vyt[0],vzt[0]
    timesteps               = tstrippy.integrator.timestamps.copy()
    # drop some points
    nskip = 4
    timesteps = timesteps[::nskip]
    xt,yt,zt = xt[::nskip],yt[::nskip],zt[::nskip]
    vxt,vyt,vzt = vxt[::nskip],vyt[::nskip],vzt[::nskip]


    eigenvalues= get_tidal_eigenvalues_along_orbit(bp, p2017pii.dimesionless_miyamoto_nagai_tidal_tensor, (xt,yt,zt,vxt,vyt,vzt))

    rt = np.sqrt(xt**2 + yt**2 + zt**2)
    Rt = np.sqrt(xt**2 + yt**2)
    # get all disc crossings
    disccrossings, _ = signal.find_peaks(-np.abs(zt))
    # get the apocenters 
    apocenters,_ = signal.find_peaks(np.abs(rt))    
    


    AXIS2 = {"xlabel": r"$\mathrm{R} [a]$","aspect": "equal","ylabel": r"$\mathrm{z} [a]$",}
    fig = plt.figure(figsize=(pagewidth,4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.1)
    axis = fig.add_subplot(gs[0, 0])
    axis2 = fig.add_subplot(gs[0, 1])


    axis2.plot(Rt,zt, color="k",linewidth=1)
    axis2.set(**AXIS2)


    axis.plot(timesteps, np.abs(eigenvalues[:,2]), label=r'$|\lambda_2|$',color="tab:blue",linewidth=2)
    axis.plot(timesteps, np.abs(eigenvalues[:,1]), label=r'$|\lambda_1|$',color="tab:red",linewidth=2,linestyle='-')
    axis.plot(timesteps, np.abs(eigenvalues[:,0]), label=r'$\lambda_0$', color="tab:green",linewidth=2)
    axis.set_yscale('log')
    axis.legend(loc='upper right', bbox_to_anchor=(.9, 1))
    axis.set_xlabel(r'$\mathrm{Time} \left[ \sqrt{\frac{a^3}{GM}}\right]$')
    axis.set_ylabel(r'$\mathrm{Tidal}\ \mathrm{Magnitude} \left[\frac{GM}{a^3}\right]$')
    axis.grid(True, which='both', linestyle='--', alpha=0.3)

    # hard set the axis ylimits from inspection
    axis.set_ylim(1e-5,5e0)

    for i in range(0,len(disccrossings),4):
        axis.text(timesteps[disccrossings[i]],np.abs(eigenvalues[disccrossings[i],2]),r'$|z|_{\mathrm{min}}$',color='k',fontsize=8)

    for i in range(0,len(apocenters),6):
        axis.text(timesteps[apocenters[i]],np.abs(eigenvalues[apocenters[i],2]) -2e-3,r'$r_{\mathrm{max}}$',color='k',fontsize=8)

    axis.set_xlim(timesteps[0],timesteps[-1])
    axis.set_title(orbitinfostring)
    return fig, axis,axis2
    # fig.savefig(outfname, dpi=300)




if __name__ == "__main__":
    # set the parameters
    apocenter = 4
    inclination = 45
    pseudo_e = 0.5
    bp = 1/5
    dtfactor = 1/1000
    Ndyntime = 4
    potentialname = 'miyamotonagai'
    initialtime = 0.0
    dx = 1/(50)


    
    # lets investigate some things 
    bps = np.logspace(-2,.5,10)

    args = []
    for bp in bps:
        args.append((apocenter, inclination, pseudo_e, bp, dtfactor, Ndyntime, potentialname, initialtime, dx))
    
    bp = 1/5
    # now lets change the inclination 
    inclinations = np.linspace(0,90,10)
    for inclination in inclinations:
        args.append((apocenter, inclination, pseudo_e, bp, dtfactor, Ndyntime, potentialname, initialtime, dx))
    # now lets change the pseudo eccentricity
    inclination= 50
    pseudo_es = np.linspace(.01,.9,10)
    dtfactor = 1/2000
    for pseudo_e in pseudo_es:
        args.append((apocenter, inclination, pseudo_e, bp, dtfactor, Ndyntime, potentialname, initialtime, dx))
    # now lets change the apocenter
    pseudo_e = 0.6
    apocenters = np.logspace(-1.5,1.5,10)
    for apocenter in apocenters:
        args.append((apocenter, inclination, pseudo_e, bp, dtfactor, Ndyntime, potentialname, initialtime, dx))


    # now lets run them in parallel 
    ncpu = mp.cpu_count() -2 
    print(f"Using {ncpu} CPUs")
    pool = mp.Pool(ncpu)
    # make the figures
    pool.starmap(make_and_save_figure, args)
    pool.close()
    pool.join()
    print("Done")