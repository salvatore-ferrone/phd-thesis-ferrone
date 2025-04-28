import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': True,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})



def simple_3D_axis(axis,limit,color='black',linewidth=1):
    axis.set_axis_off()
    axis.plot([-limit,0],[0,0],[0,0],color='black',linewidth=linewidth)
    axis.plot([0,0],[-limit,0],[0,0],color='black',linewidth=linewidth)
    axis.plot([0,0],[0,0],[-limit,0],color='black',linewidth=linewidth)
    axis.quiver(0,0,0,limit,0,0,color='black',linewidth=linewidth,arrow_length_ratio=0.1)
    axis.quiver(0,0,0,0,limit,0,color='black',linewidth=linewidth,arrow_length_ratio=0.1)
    axis.quiver(0,0,0,0,0,limit,color='black',linewidth=linewidth,arrow_length_ratio=0.1)
    axis.text(limit+limit/10,0,0,'$x$',fontsize=12)
    axis.text(0,limit+limit/10,0,'$y$',fontsize=12)
    axis.text(0,0,limit+limit/10,'$z$',fontsize=12)