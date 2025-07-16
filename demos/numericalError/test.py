import stream_time_reversability as STR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tstrippy
import datetime
import multiprocessing as mp
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Pastel2.colors)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})



# quick test to look at the time steps that are being used...
MWparams = tstrippy.Parsers.pouliasis2017pii()
staticgalaxy = ["pouliasis2017pii", MWparams ]
integrationparameters = [0, 1e-5, 5]
clusterinitialkinematics = [0, 8, 0, 220, 0, 0]
G,M,rhm,NP = MWparams[0],1e5, 5e-3,int(1e3)
aplum = tstrippy.ergodic.convertHalfMassRadiusToPlummerRadius(rhm)
hostparams = [G, M, aplum]
initialstream=tstrippy.ergodic.isotropicplummer(G,M,rhm,NP)
args = staticgalaxy, integrationparameters, clusterinitialkinematics, hostparams, initialstream
timestamps, timestamps_stream, hostorbit, streamfinal, tesc, comptimeorbit, comptimestream=STR.generate_stream_leapfrogtofinalpositions(args)