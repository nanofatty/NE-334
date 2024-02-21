from openmm import app
import openmm as mm
from openmm import unit
import sys
import mdtraj
import mdtraj.reporters
from openmm import LocalEnergyMinimizer
import numpy as np

def check_parms(parms):

    parm_error = False  # define error

    if 'ensemble' in parms.keys():
        param_list = ['steps', 'skip_steps', 'temperature', 'dt']
        for p in param_list:
            if p not in parms.keys():
                parm_error = True
    else:
        error_string = (
            "Ensemble must be present in the parameters"
            "and must have the value of NVT or NVE."
            "See the usuage instructions at the top."
        )
        print(error_string)
        return False

    if parm_error:
        error_string = '\n'.join([
            "Make sure the input parameters are:",
            "'temperature'",
            "'steps'",
            "'skip_steps'",
            "'dt'",
            "'ensemble'",
        ])
        print(error_string)
    return(parm_error)


def prepare_system(parms):

    pdb = app.PDBFile("/work/water2.pdb")
    forcefield = app.ForceField('amber10.xml', 'tip3p.xml')
    nonbonded = app.CutoffNonPeriodic

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=nonbonded,
        #nonBondedCutoff=1e3*unit.nanometer,
        rigidWater=True
    )

    if parms['ensemble'] == 'NVT':
        integrator = mm.LangevinIntegrator(parms['temperature']*unit.kelvin, 1.0/unit.picoseconds, parms['dt'])
    if parms['ensemble'] == 'NVE':
        integrator = mm.VerletIntegrator(parms['dt'])

    #Use the next line for the Reference platform, slow, easier to read, will only use 1 core
    platform = mm.Platform.getPlatformByName('Reference')
    global simulation
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.computeVirtualSites()


def minimize():
    #minimize the structure
    LocalEnergyMinimizer.minimize(simulation.context, 1e-1)


def prepare_sim(parms):
    simulation.context.setVelocitiesToTemperature(parms['temperature']*unit.kelvin)

    #Outputs progress to command line
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout,
        parms['skip_steps'],
        step=True,
        potentialEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=parms['steps'],
        separator='\t'))

    #Saves trajectory file to binary format
    global traj_filename
    traj_filename = '/work/water2_' + parms['ensemble'] + '.h5'
    simulation.reporters.append(mdtraj.reporters.HDF5Reporter(traj_filename, parms['skip_steps']))


def run_sim(parms):
    #Performs the simulation
    simulation.step(parms['steps'])
    #Close binary trajectory
    simulation.reporters[1].close()


def analyse():
    output_file = mdtraj.formats.HDF5TrajectoryFile(traj_filename)
    data = output_file.read()

    results = {
        'potE':data.potentialEnergy,
        'kinE':data.kineticEnergy,
        'positions':data.coordinates,
        'totalE':data.potentialEnergy+data.kineticEnergy,
        'time':data.time,
        'nsteps':len(data.time),
        'rOO':np.asarray([np.linalg.norm(data.coordinates[i][0] - data.coordinates[i][3]) for i in range(len(data.time))])
    }

    return(results)
