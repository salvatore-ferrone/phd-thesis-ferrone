import tstrippy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal


def f_martos(s,gamma):
    """
    An important term in the tidal tensor
    """
    numerator = gamma -1 
    denominator = 1 + s**(gamma-1)
    return 2 - numerator/denominator

def martos_mass_profile(s,gamma):
    """
    enclosed mass of the martos halo
    """
    numerator = s**gamma
    denominator = 1 + s**(gamma-1)
    return numerator/denominator

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

    mass_enclosed = martos_mass_profile(r_norm, gamma)
    f_s = f_martos(r_norm, gamma)
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


def dimesionless_miyamoto_nagai_tidal_tensor(params, xp,yp,zp):
    bp = params 
    beta = 1.0 + np.sqrt ( zp**2 + bp**2 )
    betaPrime = zp/np.sqrt(zp**2 + bp**2)
    betaPrimePrime = bp**2 / np.power(zp**2 + bp**2,3/2)
    D = np.sqrt ( xp**2 + yp**2 + beta**2 )
    T = np.zeros((3,3))
    # diagonals
    T[0,0] = 1 - 3*xp**2/D**2
    T[1,1] = 1 - 3*yp**2/D**2
    T[2,2] = betaPrime**2 + beta*betaPrimePrime - 3*(beta*betaPrime)**2 / D**2
    T[0,1] = -3*xp*yp/D**2
    T[0,2] = -3*xp*beta*betaPrime/D**2
    T[1,0] = -3*xp*yp/D**2
    T[1,2] = -3*yp*beta*betaPrime/D**2
    T[1,0] = T[0,1]
    T[2,0] = T[0,2]
    T[2,1] = T[1,2]
    T = -T/D**3
    return T

## TIDAL DEFORMATION STUFF 
def generate_unit_sphere(npoints=30):
    # 1. Generate points on a sphere
    theta = np.linspace(0, np.pi, npoints)
    phi = np.linspace(0, 2*np.pi, npoints)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x,y,z

def tidally_deform_sphere(tidal_tensor,radius,t_dyn,npoints=30):
    # make the deformed ellipsoid
    x,y,z = generate_unit_sphere(npoints)
    sphere = radius*np.array([x.flatten(), y.flatten(), z.flatten()])
    # apply the deformation 
    force = tidal_tensor @ sphere
    deformation = (1/2)*force*(t_dyn/(2*np.pi))**2
    # scale the deformation
    ellipsoid = sphere + deformation
    # reshape them
    x_deformed = ellipsoid[0].reshape(npoints,npoints)
    y_deformed = ellipsoid[1].reshape(npoints,npoints)
    z_deformed = ellipsoid[2].reshape(npoints,npoints)
    return x_deformed,y_deformed,z_deformed


### ORBIT STUFF 

def get_planar_velocity_vector(potential, params, apocenter, pseudo_e):
    """
    Sets a velocity vectors perpendicular to the radius vector and the orbital plane
    It is assumed that you are giving a position vector AT PERICENTER.
    This way, the dynamical time is the shortest time. 
    """
    # params          =   [1.0,1.0,1.0,bp] # G,M,a=1,1,1
    vel_factor      =   np.sqrt((1-pseudo_e)/(1+pseudo_e))
    # get the circular velocity
    ax,ay,az,phi    =   potential(params, apocenter[0], apocenter[1], apocenter[2])
    F               =   np.array([ax,ay,az])
    vcir            =   np.sqrt(np.linalg.norm(apocenter)*np.linalg.norm(F))
    # pick a velocity vector to be perpendicular to the radius vector and the z-axis
    unitpos         =   apocenter/np.linalg.norm(apocenter)
    v_perp          =   np.cross(unitpos, [0,0,1])
    v_perp          =   v_perp/np.linalg.norm(v_perp)
    # scale up the velocity vector
    vo              =   vel_factor*vcir*v_perp
    return vo



def get_time_info(pos,vo,dtfactor,Ndyntime,):
    """
    get the time step and number of steps
    """
    # get the time step 
    t_dyn   = 2*np.pi*np.linalg.norm(pos)/np.linalg.norm(vo)
    dt      = t_dyn*dtfactor
    integrationtime = Ndyntime*t_dyn
    # number of steps
    Nsteps = int(integrationtime/dt)
    
    return dt, Nsteps, t_dyn



def get_initial_conditions(potential,params,distance,inclination,psuedo_e):
    """
    get the initial conditions for the simulation, no polar orbits here! 
    """
    # set up the mass profile
    pos = distance*np.array([np.cos(np.radians(inclination)),0,np.sin(np.radians(inclination))])
    # get the velocity vector
    vo = get_planar_velocity_vector(potential, params, pos, psuedo_e)
    initialkinematics = np.array([pos[0],pos[1],pos[2],vo[0],vo[1],vo[2]])
    return initialkinematics




def create_local_coordinate_system(pos,vel):
    """ create a coordiante system for the orbital plane at the position of the satellite """
    # get the unit vector of the position
    e1 = -pos/np.linalg.norm(pos)
    # get the unit vector of the velocity
    # remove components of the velocity vector that are in the direction of the position vector
    vel_perp = vel - np.dot(vel, e1) * e1
    e2 = vel_perp/np.linalg.norm(vel_perp)
    # get the normal vector to the orbital plane
    e3 = np.cross(e1, e2)
    # renormalize the vectors
    e1 = e1/np.linalg.norm(e1)
    e2 = e2/np.linalg.norm(e2)
    e3 = e3/np.linalg.norm(e3)
    return e1,e2,e3

def transform_surface(matrix,xs,ys,zs):
    """
    transform the surface to the local coordinate system
    """
    x,y,z = xs.flatten(),ys.flatten(),zs.flatten()
    points= np.array([x,y,z])
    # transform the points
    transformed_points = matrix @ points
    # reshape the points
    xout= transformed_points[0,:].reshape(xs.shape)
    yout= transformed_points[1,:].reshape(ys.shape)
    zout= transformed_points[2,:].reshape(zs.shape)
    transformed_surface = np.array([xout,yout,zout])
    return transformed_surface


### PLOT STUFF 
def make_figure(pagewidth=11.75, pageheight=8.5):
    fig = plt.figure(figsize=(pagewidth-3,(pageheight-2)/2,))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig,width_ratios=[1,1],height_ratios=[1])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax3 = fig.add_subplot(gs[1], projection='3d')
    for ax in [ax1,ax3]:
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.set_axis_off()           # Hides everything: panes, ticks, and box
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
    ax2.xaxis.pane.set_visible(False)
    ax2.yaxis.pane.set_visible(False)
    ax2.zaxis.pane.set_visible(False)
    ax2.set_axis_off()           # Hides everything: panes, ticks, and box
    ax2.set_facecolor("none")        
    return fig, ax1, ax2, ax3


def zero_crossing_segments_indices(z):
    """
    Returns:
        segments: list of index arrays for each segment between zero crossings
        zpos: list of index arrays where z > 0
        zneg: list of index arrays where z < 0
    """
    z = np.asarray(z)
    sign_changes = np.where(np.diff(np.sign(z)) != 0)[0] + 1
    indices = np.concatenate(([0], sign_changes, [len(z)]))
    segments = [np.arange(indices[i], indices[i+1]) for i in range(len(indices)-1)]
    zpos = [seg for seg in segments if np.all(z[seg] > 0)]
    zneg = [seg for seg in segments if np.all(z[seg] < 0)]
    return segments, zpos, zneg

def do_axis1(ax1,LIMS,pos_local,plane,orbit_segment,basis_vectors,AXISPROPERTIES):
    # unpack the arguments 
    X_plane, Y_plane, Z_plane = plane
    xt,yt,zt = orbit_segment
    e1,e2,e3 = basis_vectors
    
    # get the zero crossings 
    _, zpos, zneg = zero_crossing_segments_indices(zt)
    for i in range(len(zpos)):
        ax1.plot(xt[zpos[i]], yt[zpos[i]], zt[zpos[i]], lw=1,color='black', alpha=1)
    for i in range(len(zneg)):
        ax1.plot(xt[zneg[i]], yt[zneg[i]], zt[zneg[i]], lw=0.5,color='k', alpha=0.5)

    ax1.plot(LIMS,[0,0], [0,0], color='k', lw=1)
    ax1.plot([0,0],LIMS,[0,0], color='k', lw=1)
    ax1.plot([0,0], [0,0], LIMS, color='k', lw=1)
    lenaxis = np.max(LIMS)
    # make the basis vectors equal to 1/4 of the axis length
    e1 = (lenaxis/2)*e1
    e2 = (lenaxis/2)*e2
    e3 = (lenaxis/2)*e3
    # add the new basis vectors to ax1
    ax1.quiver(pos_local[0], pos_local[1], pos_local[2], e1[0], e1[1], e1[2], color='black',)
    ax1.quiver(pos_local[0], pos_local[1], pos_local[2], e2[0], e2[1], e2[2], color='blue',)
    ax1.quiver(pos_local[0], pos_local[1], pos_local[2], e3[0], e3[1], e3[2], color='green')

    # add the labels 
    labelmax = np.max(LIMS)
    little_move = labelmax/20
    ax1.text(labelmax,-little_move,little_move,r'$\hat{x}$',  fontsize="small")
    ax1.text(0,labelmax,little_move,r'$\hat{y}$', fontsize="small")
    ax1.text(0,little_move,labelmax,r'$\hat{z}$', fontsize="small")


    ax1.plot_surface(X_plane, Y_plane, Z_plane, color='lightblue', alpha=0.3, 
                    edgecolor='lightblue', linewidth=0.5)
    ax1.set(**AXISPROPERTIES)
    return ax1


def do_axis2(ax2,elipsoid,factors,eigenvectors,view_init=(30,30)):
    # unpack
    x_ell,y_ell,z_ell = elipsoid
    little_factor,axislen,little_move = factors
    eigen1,eigen2,eigen3 = eigenvectors
    
    ax2.plot_surface(x_ell, y_ell, z_ell, color='r', alpha=.9)

    ax2.plot(np.array([0,0]),np.array([0,0]),np.array([0,axislen]), color='k', lw=2)
    ax2.plot(np.array([0,0]),np.array([0,axislen]),np.array([0,0]), color='k', lw=2)
    ax2.plot(np.array([0,axislen]),np.array([0,0]),np.array([0,0]), color='k', lw=2)

    ax2.text(axislen,-little_move,little_move,r'$\hat{x}$',  fontsize="small")
    ax2.text(0,axislen,little_move,r'$\hat{y}$', fontsize="small")
    ax2.text(0,little_move,axislen,r'$\hat{z}$', fontsize="small")

    ax2.quiver(0,0,0,eigen1[0],eigen1[1],eigen1[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)
    ax2.quiver(0,0,0,eigen2[0],eigen2[1],eigen2[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)
    ax2.quiver(0,0,0,eigen3[0],eigen3[1],eigen3[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)

    # make the vectors go in both directions
    ax2.quiver(0,0,0,-eigen1[0],-eigen1[1],-eigen1[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)
    ax2.quiver(0,0,0,-eigen2[0],-eigen2[1],-eigen2[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)
    ax2.quiver(0,0,0,-eigen3[0],-eigen3[1],-eigen3[2], length=axislen, color='tab:red', arrow_length_ratio=0.1)
    ax2.set_xlim([-axislen,axislen])
    ax2.set_ylim([-axislen,axislen])
    ax2.set_zlim([-axislen,axislen])
    ax2.set_aspect('equal')
    ax2.autoscale(enable=False)
    ax2.view_init(*view_init)
    return ax2

def do_axis3(ax3,ellipsoid_rot,factors,eigenvectors_rot,view_init=(30,30)):
    # unpack
    eigen1,eigen2,eigen3 = eigenvectors_rot
    little_factor,axislen,little_move = factors

    ax3.quiver(0,0,0,eigen1[0],eigen1[1],eigen1[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0,eigen2[0],eigen2[1],eigen2[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0,eigen3[0],eigen3[1],eigen3[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)

    # make the vectors go in both directions
    ax3.quiver(0,0,0,-eigen1[0],-eigen1[1],-eigen1[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0,-eigen2[0],-eigen2[1],-eigen2[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0,-eigen3[0],-eigen3[1],-eigen3[2], length=2*axislen/3, color='tab:red', arrow_length_ratio=0.1)
    ax3.plot_surface(ellipsoid_rot[0], ellipsoid_rot[1], ellipsoid_rot[2], color='r', alpha=1)

    ax3.plot(np.array([0,0]),np.array([0,0]),np.array([0,axislen]), color='green', lw=2)
    ax3.plot(np.array([0,0]),np.array([0,axislen]),np.array([0,0]), color='blue', lw=2)
    ax3.plot(np.array([0,axislen]),np.array([0,0]),np.array([0,0]), color='black', lw=2)

    ax3.text(axislen,-little_move,little_move,r'-$\hat{r}$',  fontsize="small")
    ax3.text(0,axislen,little_move, r'$\hat{v}$', fontsize="small")
    ax3.text(0,little_move,axislen,r'-$\hat{L}$', fontsize="small")

    ax3.set_xlim([-axislen,axislen])
    ax3.set_ylim([-axislen,axislen])    
    ax3.set_zlim([-axislen,axislen])
    ax3.view_init(*view_init)
    ax3.set_aspect('equal')
    return ax3


def make_frame(index,params,orbit,dx,t_dyn,factors,AXIS,LIMS,plane,tail_indexes=20):
    """
    """
    bp = params
    xt,yt,zt,vxt,vyt,vzt = orbit
    little_factor,axislen,little_move = factors
    pos_local = np.array([xt[index],yt[index],zt[index]])
    vel_local = np.array([vxt[index],vyt[index],vzt[index]])
    # get the tidal tensor 
    tidal_tensor=dimesionless_miyamoto_nagai_tidal_tensor(bp,pos_local[0],pos_local[1],pos_local[2])    
    # get orbit indexes 
    upindex = index
    down_index = index-tail_indexes if index-tail_indexes > 0 else 0
    # pack the orbit segment
    orbit_segment = xt[down_index:upindex],yt[down_index:upindex],zt[down_index:upindex]
    # get the deformed sphere
    x_ell,y_ell,z_ell = tidally_deform_sphere(tidal_tensor,dx,t_dyn/2)
    ellipsoid= (x_ell,y_ell,z_ell)
    # get the eigen vectots for the elipse's principal axes
    eigen           =   np.linalg.eig(tidal_tensor)
    eigenvectors    =   eigen.eigenvectors
    eigen1 = eigenvectors[:,0] # having them packed into the matrix was confusing
    eigen2 = eigenvectors[:,1]
    eigen3 = eigenvectors[:,2]    

    # get the unit vectors for the local coordinates system 
    e1,e2,e3= create_local_coordinate_system(pos_local,vel_local)
    basis_vectors= (e1,e2,e3)
    # make the new coordiante system 
    R               = np.array([e1,e2,e3])
    # put the ellipsoid in the new coordinate system
    ellipsoid_rot   =transform_surface(R, x_ell,y_ell,z_ell)
    eigen1_rotated = np.dot(R, eigen1)
    eigen2_rotated = np.dot(R, eigen2)
    eigen3_rotated = np.dot(R, eigen3)    
    # make sure we're always parallel with e1 and not anti parallel
    eigen1_rotated = -eigen1_rotated if np.dot(eigen1_rotated,e1) < 0 else eigen1_rotated
    
    # compute the axim and elev for the other axis
    # azim = (180/np.pi)*np.arctan2(pos_local[0],pos_local[1])
    # roll = (180/np.pi)*np.acos(np.dot([0,0,1],e3))
    # elev = 30
    elev,azim,roll = 30,30,0


    fig, ax1, ax2,ax3 = make_figure()
    # # add the orbit 
    do_axis1(ax1,LIMS,pos_local,plane,orbit_segment,basis_vectors,AXIS)
    do_axis2(ax2,ellipsoid,factors,(eigen1,eigen2,eigen3),view_init=(30,30))
    do_axis3(ax3,ellipsoid_rot,factors,(eigen1_rotated,eigen2_rotated,eigen3_rotated),view_init=(30,30))
    
    return fig, ax1, ax2, ax3
    # fig.savefig(fname=fname, dpi=300)
    # plt.close(fig)


def make_and_save_frame(index,params,orbit,dx,t_dyn,factors,AXIS,LIMS,plane,tail_indexes=20):
    """
    """

    fname = "../frames/frame_%04d.png" % index
    fig, ax1, ax2, ax3 = make_frame(index,params,orbit,dx,t_dyn,factors,AXIS,LIMS,plane,tail_indexes)
    fig.savefig(fname=fname, dpi=300)
    plt.close(fig)
    return None