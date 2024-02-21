from __future__ import print_function
from openmm import app
import openmm as mm
from openmm import unit
from sys import stdout
from openmmtools import testsystems
import mdtraj

def check_parms(parms):
    parm_error = False
    for i in ['steps', 'skip_steps', 'temperature', 'equil_steps','N','density']:
        if i not in parms.keys():
            parm_error = True

    if parm_error:
        print('Make sure the input parameters are:')
        print("'steps', 'skip_steps', 'temperature', 'equil_steps','N','density'")

    return(parm_error)

def prepare_system(parms):

    #test_sys = testsystems.LennardJonesFluid(nparticles=200, reduced_density=0.50)
    test_sys = testsystems.LennardJonesFluid(
        nparticles=parms['N'],
        mass=39.9*unit.dalton,
        sigma=3.4*unit.angstrom,
        epsilon=0.238*unit.kilocalories_per_mole,
        cutoff=None,
        reduced_density=parms['density']
    )

    (system, positions) = test_sys.system, test_sys.positions

    #integrator = mm.VerletIntegrator(.01 * unit.femtoseconds)
    integrator = mm.LangevinIntegrator(
        parms['temperature']*unit.kelvin,
        1.0/unit.picoseconds,
        1.0*unit.femtoseconds
    )

    platform = mm.Platform.getPlatformByName('Reference')

    global simulation
    simulation = app.Simulation(test_sys.topology, system, integrator, platform)
    simulation.context.setPositions(test_sys.positions)

    print('The size of the periodic box is: ', system.getDefaultPeriodicBoxVectors())

def minimize():
    simulation.minimizeEnergy()
    print('Minimizing...')

def equilibrate(parms):
    print('Initializing velocities to Boltzmann distribution')
    simulation.context.setVelocitiesToTemperature(parms['temperature']*unit.kelvin)

    print('Equilibrating...')
    simulation.step(parms['equil_steps']*parms['skip_steps'])

def prepare_sim(parms):
    simulation.reporters.append(app.StateDataReporter(
        stdout, parms['skip_steps'], step=True,
        potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
        speed=True, totalSteps=parms['steps'], separator='\t'
    ))

    global ar_traj
    ar_traj = '/work/ar_liquid_traj' + str(parms['N']) + '.h5'
    # simulation.reporters.append(
    #     app.PDBReporter(ar_traj + '.pdb', parms['skip_steps'])
    # )
    simulation.reporters.append(
        mdtraj.reporters.HDF5Reporter(ar_traj, parms['skip_steps'])
    )

def run_sim(parms):
    print('Simulation beginning...')
    simulation.step(parms['steps']*parms['skip_steps'])
    simulation.reporters[1].close()

def v_analysis():

    import numpy as np

    # trajectory file
    output_file = mdtraj.formats.HDF5TrajectoryFile(ar_traj)
    data = output_file.read()

    results = {
        'potE':data.potentialEnergy,
        'time':data.time,
        'nsteps':len(data.time),
        'mean':np.mean(data.potentialEnergy),
        'variance':np.var(data.potentialEnergy)
    }

    return(results)



import numpy as np

def gen_pair_dist():

    input_data = mdtraj.load(ar_traj)
    ##############################
    # parameters
    nbins=100
    rmin=0.1
    #rmax=2.0594033385430914/2.
    rmax=input_data.unitcell_lengths[0,0]/2.
    dr=(rmax-rmin)/float(nbins)
    #volume=(rmax*2.)**3
    volume=input_data.unitcell_lengths[0,0]*input_data.unitcell_lengths[0,1]*input_data.unitcell_lengths[0,2]

    #############################


    N = int(input_data.n_atoms)
    Nsteps=input_data.n_frames
    print('There are ', N, ' Argon atoms in the trajectory')
    pairs = []
    for i in range(N):
        for j in range(i+1,N):
            pairs.append([i,j])
    print('There are ', len(pairs), ' Ar-Ar pairs')

    distances = mdtraj.compute_distances(input_data,pairs)

    print('There are ', len(distances), ' steps in the trajectory')

    histo=np.zeros(nbins,float)
    #accumulate histograms
    n_count=0
    for ArAr in distances:
        for d in ArAr:
            index=int(np.floor((d-rmin)/dr))
            if index < nbins:
                histo[index]+=1.
    #normalize histogram and divide by jacobian
    for i in range(nbins):
        r=rmin+i*dr
        histo[i]=histo[i]/(2.*np.pi*r*r*dr*N*N/volume)/float(Nsteps)

    Ar_file=open('/work/Ar_histo','w')
    NN=0.
    for i in range(nbins):
        r=rmin+i*dr
        Ar_file.write(str(rmin+i*dr)+' '+str(histo[i])+'\n')
        if (r<0.5):
            NN+=histo[i]*r*r
    print('N = ', N,'V =',volume,'Delta r =',r)
    print('Number of neighbours = ',dr*4.*np.pi*(N/volume)*NN)
    Ar_file.close()