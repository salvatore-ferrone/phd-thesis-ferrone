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
    """
    An important term in the tidal tensor
    """
    numerator = gamma -1 
    denominator = 1 + s**(gamma-1)
    return 2 - numerator/denominator

def mass_profile(s,gamma):
    """
    enclosed mass of the martos halo
    """
    numerator = s**gamma
    denominator = 1 + s**(gamma-1)
    return numerator/denominator

def compute_vcirc(haloparams, x0):
    """ 
    the circular velocity of the halo at a given radius
    """
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
    circle = np.vstack((x,y,z))
    # get the tidal force at each point
    dF_circ = np.dot(tensor, circle)
    # get the characteristic time to scale the force into distance 
    r0_mag = np.linalg.norm(r0)
    vcirc=compute_vcirc(haloparams,r0_mag)
    t_orbit = 2*np.pi*r0_mag/vcirc
    dtstep = t_orbit/dtfactor
    myellipse = circle + (1/2)*dF_circ*dtstep**2
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
    datasize= datasize.to(u.Gbyte)  # Convert to Gbyte
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
    # xpt, ypt, zpt, vptx, vpty, vptz = tstrippy.integrator.leapfrogintime(NSTEP,NP)
    tstrippy.integrator.leapfrogtofinalpositions()
    xf,yf,zf = tstrippy.integrator.xf.copy(),tstrippy.integrator.yf.copy(),tstrippy.integrator.zf.copy()
    vxf,vyf,vzf=tstrippy.integrator.vxf.copy(),tstrippy.integrator.vyf.copy(),tstrippy.integrator.vzf.copy()
    tstrippy.integrator.deallocate()
    endtime=datetime.datetime.now()
    print("Time taken to integrate the orbit: ", endtime-starttime)
    stream = (xf,yf,zf,vxf,vyf,vzf)
    # get the host orbit
    hostorbit = (xt, yt, zt, vxt, vyt, vzt)
    
    return timesteps,  hostorbit, stream


def get_tidal_vectors(pos,dx,gamma, n_vectors=25):
    dxs,dys = np.linspace(-dx,dx,n_vectors), np.linspace(-dx,dx,n_vectors)
    dXs, dYs = np.meshgrid(dxs,dys)
    dXs,dYs  = dXs.flatten(), dYs.flatten()
    dZs = np.zeros_like(dXs)
    tidal_tensor = normalized_martos_tidal_tensor(pos,gamma)
    F_tides = tidal_tensor @ np.array([dXs,dYs,dZs])
    return dXs, dYs, F_tides


def unscale(r_scale,v_scale, hostorbit,stream):
    # unpack the orbit
    xt,yt,zt,vx,vy,vz=hostorbit
    xpt,ypt,zpt,vxpt,vypt,vzpt=stream
    # scale everything 
    xt_= xt/r_scale.value
    yt_= yt/r_scale.value
    zt_= zt/r_scale.value
    
    xt_= xt/r_scale.value
    yt_= yt/r_scale.value
    zt_= zt/r_scale.value
    vxt_= vx/v_scale.value
    vyt_= vy/v_scale.value
    vzt_= vz/v_scale.value
    # unscale everything 
    xpt_ = xpt/r_scale.value
    ypt_ = ypt/r_scale.value
    zpt_ = zpt/r_scale.value
    vxpt_ = vxpt/v_scale.value
    vypt_ = vypt/v_scale.value
    vzpt_ = vzpt/v_scale.value
    
    orbit = xt_, yt_, zt_, vxt_, vyt_, vzt_
    stream = xpt_, ypt_, zpt_, vxpt_, vypt_, vzpt_
    return orbit, stream

def plot_martos_tidal_field(orbitxy,streamxy,tidal_stuff,ellipseCirlcle, cbarstuff,
                       AXES1 = {"xlabel": r"$x$ [$r_h$]","ylabel": r"$y$ [$r_h$]","aspect": "equal",},
                       AXES2 = {"xlabel": r"$x$ [$r_h$]","aspect": "equal",}):
    # unpack the inputs
    xt_, yt_ = orbitxy
    dXs, dYs, F_tides = tidal_stuff
    xp,yp= streamxy
    myellipse, circle = ellipseCirlcle
    norm,cmap= cbarstuff

    F_tides_mag = np.linalg.norm(F_tides,axis=0)
    F_colors = cmap(norm(F_tides_mag))

    pos = np.array([xt_[-1],yt_[-1]])
    # add the limits to the AXES2
    AXES2["xlim"] = [pos[0]+dXs.min(), pos[0]+dXs.max()]
    AXES2["ylim"] = [pos[1]+dYs.min(), pos[1]+dYs.max()]
    # set the limits to the AXES1
    dw = .01
    AXES1["xlim"] = [(1+dw)*xt_.min(), (1+dw)*xt_.max()]
    AXES1["ylim"] = [(1+dw)*yt_.min(), (1+dw)*yt_.max()]

    # set up figure 
    fig=plt.figure(figsize=(11.75-2,5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.30)
    axes = [ ]
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    caxis=fig.add_subplot(gs[0, 2])

    axes[0].plot(xt_, yt_, color='gray', lw=0.5,)
    axes[0].scatter(xp,yp, color='black', s=10, )


    axes[1].scatter(xp,yp, color='k', s=2, zorder=0)
    axes[1].plot(pos[0]+myellipse[0], pos[1]+myellipse[1], color='black', lw=1, label='Tidal deformation')
    axes[1].plot(pos[0]+circle[0], pos[1]+circle[1], color='k', lw=1, linestyle=":")
    axes[1].quiver(pos[0]+dXs, pos[1]+dYs, F_tides[0]/F_tides_mag, F_tides[1]/F_tides_mag,color=F_colors,scale=30, width=1/200,)

    # get the unit vector of the position 
    unitPos = pos/np.linalg.norm(pos)
    # make the unit vector the a resonable size 
    len1 = (xt_.max()-xt_.min())/6
    len2 = (dXs.max()-dXs.min())/6
    axes[0].quiver(pos[0], pos[1], -len1*unitPos[0], -len1*unitPos[1],  color="k", units='xy', scale=1)
    axes[1].quiver(pos[0], pos[1], -len2*unitPos[0], -len2*unitPos[1],  color="k", units='xy', scale=1)

    axes[0].set(**AXES1)
    axes[1].set(**AXES2)
    axes[1].legend(frameon=False)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=caxis)
    cbar.set_label(r"$\left| \mathbf{F}_{\mathrm{tides}} \right|$", fontsize=12)
    caxis.tick_params(labelsize=12)
    caxis.yaxis.set_label_position("left")

    # Make the colorbar the same height as axes[1]
    pos_ax1 = axes[1].get_position()
    pos_cax = caxis.get_position()
    caxis.set_position([pos_cax.x0, pos_ax1.y0, pos_cax.width, pos_ax1.height])
    return fig, axes



def compute_and_make_plot(haloparams, isotropicplummer, orbitparams, norm, cmap, dx, r_scale, v_scale, n_vectors=25, dtfactor=10):
    # unpack the params
    G,halomass,haloradius,gamma,rcut = haloparams
    G,cluster_mass,cluster_radius,NP = isotropicplummer
    r0,eccen,num_orbits = orbitparams


    figtitle = r"$\gamma, e, r_o$ = ({:.02f},{:.02f},{:.02f})".format(gamma,eccen,r0/r_scale.value)
    fname = "../../images/martos_tidal_field_{:d}_{:d}_{:d}.png".format(int(100*gamma),int(eccen*100),int(100*r0/r_scale.value))

    # do the simulation
    timesteps,  hostorbit, freshstream=make_stream(haloparams,orbitparams, isotropicplummer)
    # unscale the result 
    orbit,stream= unscale(r_scale,v_scale, hostorbit,freshstream)
    # get the position of the host
    xt, yt, zt, vxt, vyt, vzt = orbit
    xp, yp, zp, vxp, vyp, vzp = stream
    pos = np.array([xt[-1],yt[-1],zt[-1]])
    myellipse,circle=get_dimensionless_deformed_ellipse(haloradius*pos, dx/2,haloparams,dtfactor=dtfactor)
    dXs, dYs, F_tides= get_tidal_vectors(pos,dx,gamma, n_vectors)

    # pack up the stream and orbit
    orbitxy = xt, yt
    streamxy = xp, yp
    tidal_stuff = dXs, dYs, F_tides
    ellipseCirlcle = myellipse, circle
    cbarstuff = norm,cmap
    # make the fig 
    fig, axes = plot_martos_tidal_field(orbitxy, streamxy, tidal_stuff,
                                ellipseCirlcle, cbarstuff)
    axes[0].set_title(figtitle, fontsize="large", y=1)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print("Saved figure to {}".format(fname))
    plt.close(fig)
    return None