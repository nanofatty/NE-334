import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
from openmmtools import testsystems
from sys import stdout
import mdtraj

def check_parms(parms):
    parm_error = False
    for i in ['steps', 'skip_steps', 'temperature', 'equil_steps','N','Box_edge']:
        if i not in parms.keys():
            parm_error = True

    if parm_error:
        print('Make sure the input parameters are:')
        print("'steps', 'skip_steps', 'temperature', 'equil_steps','N','Box_edge'")

    return(parm_error)

def prepare_system(parms):

    test_sys = testsystems.WaterBox(box_edge=parms['Box_edge'], cutoff=parms['Box_edge']/2.)
    (system, positions) = test_sys.system, test_sys.positions

    print('The size of the periodic box is: ', system.getDefaultPeriodicBoxVectors())

    integrator = mm.LangevinIntegrator(parms['temperature']*unit.kelvin, 1.0/unit.picoseconds,1.0*unit.femtoseconds)

    platform = mm.Platform.getPlatformByName('Reference')
    platform = mm.Platform.getPlatformByName('CPU')

    global simulation
    simulation = app.Simulation(test_sys.topology, system, integrator, platform)
    simulation.context.setPositions(test_sys.positions)

def minimize():
    print('Minimizing...')
    simulation.minimizeEnergy()

def equilibrate(parms):
    print('Initializing velocities to Boltzmann distribution')
    simulation.context.setVelocitiesToTemperature(parms['temperature']*unit.kelvin)

    print('Equilibrating...')
    simulation.step(parms['equil_steps']*parms['skip_steps'])

def prepare_sim(parms):
    simulation.reporters.append(app.StateDataReporter(stdout, parms['skip_steps'], step=True,
        potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
        speed=True, totalSteps=parms['steps'], separator='\t'))

    #simulation.reporters.append(app.PDBReporter('h2o_liquid_traj.pdb', parms['skip_steps']))
    global waterbox_traj
    waterbox_traj = '/work/h2o_liquid_traj.h5'
    simulation.reporters.append(mdtraj.reporters.HDF5Reporter(waterbox_traj, parms['skip_steps']))

def run_sim(parms):
    print('Simulation beginning...')
    simulation.step(parms['steps']*parms['skip_steps'])
    simulation.reporters[1].close()

def gen_pair_dist():

    import numpy as np

    # trajectory file
    input_data = mdtraj.load(waterbox_traj)

    ##############################
    # parameters
    nbins=100
    rmin=0.1
    rmax=input_data.unitcell_lengths[0,0]/2.
    dr=(rmax-rmin)/float(nbins)
    volume=input_data.unitcell_lengths[0,0]*input_data.unitcell_lengths[0,1]*input_data.unitcell_lengths[0,2]

    #############################

    N = int(input_data.n_atoms/3)
    Nsteps=input_data.n_frames

    print('There are ', N, ' waters in the trajectory')
    OO_pairs = []
    OH_pairs = []
    for i in range(N):
        for j in range(i+1,N):
            OO_pairs.append([i*3,j*3])

            OH_pairs.append([i*3,j*3+1])
            OH_pairs.append([i*3,j*3+2])
            OH_pairs.append([j*3,i*3+1])
            OH_pairs.append([j*3,i*3+2])

    print('There are ', len(OO_pairs), ' O-O pairs')
    print('There are ', len(OH_pairs), ' O-H pairs')

    OO_distances = mdtraj.compute_distances(input_data,OO_pairs)
    OH_distances = mdtraj.compute_distances(input_data,OH_pairs)

    print('There are ', len(OO_distances), ' steps in the trajectory')
    print('There are ', len(OH_distances), ' steps in the trajectory')

    OO_histo=np.zeros(nbins,float)
    #accumulate histograms
    for OO in OO_distances:
        for d in OO:
            index_OO=int(np.floor((d-rmin)/dr))
            if index_OO < nbins:
                OO_histo[index_OO]+=1.
    OH_histo=np.zeros(nbins,float)
    #accumulate histograms
    for OH in OH_distances:
        for d in OH:
            index_OH=int(np.floor((d-rmin)/dr))
            if index_OH < nbins:
                OH_histo[index_OH]+=1.

    #normalize histogram and divide by jacobian
    for i in range(nbins):
            r=rmin+i*dr
            OO_histo[i]=OO_histo[i]/(2.*np.pi*r*r*dr*N*N/volume)/float(Nsteps)
            OH_histo[i]=OH_histo[i]//(2.*np.pi*r*r*dr*N*N*4./volume)/float(Nsteps)

    OO_file=open('/work/OO_histo','w')
    for i in range(nbins):
        OO_file.write(str(rmin+i*dr)+' '+str(OO_histo[i])+'\n')
    OO_file.close()
    OH_file=open('/work/OH_histo','w')
    for i in range(nbins):
        OH_file.write(str(rmin+i*dr)+' '+str(OH_histo[i])+'\n')
    OH_file.close()