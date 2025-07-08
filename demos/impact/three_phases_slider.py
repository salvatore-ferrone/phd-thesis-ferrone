import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import gridspec

def get_kick_velocity(PI1,M,b,alpha,y,Wperp,Wpar,Rs):
    """ 
    the changes in velocoity of the particle due to the gravitational force of da passage
    """
    Wmag=np.sqrt(Wperp**2+Wpar**2)
    term1 = (b*Wmag**2*np.cos(alpha))
    term2 = y*Wperp*Wpar*np.sin(alpha)
    term3 = (b**2 + Rs**2)*Wmag**2
    term4 = (Wperp**2)*y**2

    denominator = Wmag*(term3+term4)

    dX= 2*PI1*M*(term1+term2)/denominator
    dY=-2*PI1*(M*(Wperp**2)*y)/denominator
    # re-arrange the numerator for dZ    
    term1= (b*Wmag**2*np.sin(alpha))
    term2= y*Wperp*Wpar*np.cos(alpha)
    dZ= 2*PI1*M*(term1-term2)/denominator
    return np.array([dX,dY,dZ])


def plot_three_phases(axis,b, alpha, Wperp, Wpar):
    # make vector per impact parameter
    B = np.array([b*np.cos(alpha),0,b*np.sin(alpha)])

    # Define the line for the perturber
    W=np.array([-Wperp*np.sin(alpha),Wpar,Wperp*np.cos(alpha)])
    Wnorm=W/np.linalg.norm(W)
    
    # find where the line intersects the x-y plane for clarity
    t_inter = -B[2]/Wnorm[2]

    # make time array for the line
    t = np.linspace(-t_inter,t_inter,10)
    intersection_point=np.array([Wnorm[0]*t_inter + B[0],Wnorm[1]*t_inter+B[1],Wnorm[2]*t_inter+B[2]])    
    # make a parametric line for W
    Wline = np.array([Wnorm[0]*t + B[0],Wnorm[1]*t+B[1],Wnorm[2]*t+B[2]])

    # find the position of the particle at that time
    
    # Define the parameters for the arc
    radius = np.linalg.norm(B)/2
    theta = np.linspace(0, alpha, 100)

    # Parametric equations for the 3D arc
    x_arc = radius * np.cos(theta)
    y_arc = radius * np.sin(theta) * np.cos(np.arccos(B[1] / np.linalg.norm(B[1:])))
    z_arc = radius * np.sin(theta) * np.sin(np.arccos(B[1] / np.linalg.norm(B[1:])))



    limit=1
    # Plot the X, Y, and Z coordinate axes
    axis.plot3D([0, limit], [0, 0], [0, 0], 'k',)
    axis.plot3D([0, 0], [0, limit], [0, 0], 'k',)
    axis.plot3D([0, 0], [0, 0], [0, limit], 'k',)

    # Add x-y-z axis labels
    axis.text(limit, 0, 0, 'X', color='k')
    axis.text(0, limit, 0, 'Y', color='k')
    axis.text(0, 0, limit, 'Z', color='k')

    # Plot the vector B
    B_handle = axis.quiver(0, 0, 0, B[0], B[1], B[2], color='r', label='B')

    # Add a label for the vector B
    axis.text(B[0], B[1], B[2], 'b', color='r')

    arc_handle = axis.plot(x_arc,y_arc,z_arc)

    axis.text(x_arc.mean(),y_arc.mean(),z_arc.mean(),r'$\alpha$',color='b')

    # plot the line W
    Wline_handle = axis.plot(Wline[0],Wline[1],Wline[2],'b')


    # Calculate the intersection point of the line W with the xy-plane

    # Plot the line from the intersection point to the xy-plane
    axis.plot3D([intersection_point[0], intersection_point[0]], [0, intersection_point[1]], [0, 0], 'k--')

    # Update the plot
    # plt.show()
    
    axis.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='k')
    # ax.plot([intersection_point[0],intersection_point[0]],[0,0],[0,intersection_point[2]],'k--')

    # Turn off the axis panes
    axis.axis('off')

    # Return the plot handles
    return axis, B_handle, arc_handle, Wline_handle


def do_plot_all(PI1,M,b,alpha,Wperp,Wpar,Rs,ylims=(-50,50),xlims=(-1,1)):
    y=np.linspace(-50,50,1000)
    dV=get_kick_velocity(PI1,M,b,alpha,y,Wperp,Wpar,Rs)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0)

    # Create a new figure
    fig = plt.figure(figsize=(15, 4))

    # Add two 3D subplots to the figure using the grid layout
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], projection='3d')

    non_dimen_text = r"$\Pi$: $\frac{{GM_c}}{{V_c^2 R_c}}$ = {:2.1f}".format(PI1)
    ax2.text(-1.5, 0.5,0, non_dimen_text, fontsize=18, ha='center')

    plot_three_phases(ax2, b, alpha, Wperp, Wpar)
    ax1.plot(y, dV[0], label='dVx')
    ax1.plot(y, dV[1], label='dVy')
    ax1.plot(y, dV[2], label='dVz')
    ax1.legend()
    ax1.set_xlabel('y [$x_c$]')
    ax1.set_ylabel('dV [$v_c$]')

    title_text=r"b:{:2.2f},$\alpha$:{:2.0f}$^{{\circ}}$, $W_\bot$:{:2.1f} $W_\parallel${:2.1f}".format(b,(180/np.pi)*alpha,Wperp,Wpar)
    ax1.set_title(title_text)

    figname="pi_{:2.1f}_b_{:2.2f}_alpha_{:2.0f}_Wperp_{:2.1f}_Wpar_{:2.1f}.png".format(PI1,b,(180/np.pi)*alpha,Wperp,Wpar)
    
    ax1.set_ylim(ylims)
    ax1.set_xlim(xlims)
    
    return fig,figname




