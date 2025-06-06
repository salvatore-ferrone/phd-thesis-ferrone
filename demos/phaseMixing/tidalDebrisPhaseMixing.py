import tstrippy
def integrate_particles(staticgalaxy,integrationparams,initialkinematics,hostperturber):
    NSTEP = int(integrationparams[2])
    NPARTICLES = len(initialkinematics[0])
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setstaticgalaxy(*staticgalaxy)
    tstrippy.integrator.setinitialkinematics(*initialkinematics)
    tstrippy.integrator.setintegrationparameters(*integrationparams)
    tstrippy.integrator.inithostperturber(*hostperturber)
    xt,yt,zt,vxt,vyt,vzt=tstrippy.integrator.leapfrogintime(NSTEP,NPARTICLES)
    timestamps = tstrippy.integrator.timestamps.copy() 
    tstrippy.integrator.deallocate()
    return timestamps,xt,yt,zt,vxt,vyt,vzt