import mlatom as ml

odm2 = ml.models.methods(method='ODM2', read_keywords_from_file='mndokw_fulvene')
model = ml.models.msani(model_file='fulvene_delta_model.pt', nstates=2)



class delta_model_namd():
    def __init__(self, qm_method, ml_model):
        self.qm_method = qm_method
        self.ml_model = ml_model
    def predict(self,molecule=None,nstates=5, current_state=0, calculate_energy=True, calculate_energy_gradients=True):

        self.ml_model.predict(molecule=molecule, calculate_energy_gradients=[True]*nstates, calculate_energy=True, current_state=current_state, nstates=nstates)
        tempmol = ml.data.molecule()
        tempmol.atoms = molecule.atoms
        self.qm_method.predict(molecule=tempmol, calculate_energy=True, nstates=nstates, current_state=0, calculate_energy_gradients=[True]*nstates)

        for istate in range(nstates):
            molecule.electronic_states[istate].energy += tempmol.electronic_states[istate].energy
            molecule.electronic_states[istate].energy_gradients += tempmol.electronic_states[istate].energy_gradients

        molecule.energy = molecule.electronic_states[current_state].energy
        molecule.energy_gradients = molecule.electronic_states[current_state].energy_gradients

delta_model = delta_model_namd(model, odm2)
init_cond_db = ml.data.molecular_database.load('init_cond_db_fulvene.json', format='json')
namd_kwargs = {
            'model': delta_model,
            'time_step': 0.1, # fs
            'maximum_propagation_time': 60,
            'dump_trajectory_interval': 30,
            'filename':"traj.h5",
            'format':"h5md",
            'hopping_algorithm': 'LZBL',
            'initial_state':1,
            'nstates': 2,
            'reduce_kinetic_energy': True,
            'reduce_memory_usage':True
            }

dyns = ml.simulations.run_in_parallel(molecular_database=init_cond_db[:1000], task=ml.namd.surface_hopping_md, task_kwargs=namd_kwargs, create_and_keep_temp_directories=True, proceed_on_error=True)
