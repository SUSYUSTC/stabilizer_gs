import numpy as np
from state import Hamiltonian, StateMachinePeriodic, generate_Sright_periodic

max_float = 1e100
cmax = 10  # memory amount to malloc, should >= c, otherwise may cause MemoryError
H_periodic = Hamiltonian.from_file_periodic("../hamiltonian_periodic.txt", cmax)
c = 3  # supercell size


def periodic_evolve_states(states, clear_history=True, c=1):
    state_dict = {hash(state): state for state in states}
    state_periodic.state_dict = state_dict
    state_periodic.m = H_periodic.k + H_periodic.l - 1
    for m in range(H_periodic.l * c):
        state_periodic.evolve()
    states = [state.shift(-H_periodic.l * c, clear_history=clear_history) for state in state_periodic.state_dict.values()]
    return states


Sright_periodic = generate_Sright_periodic(H_periodic, cmax)
state_periodic = StateMachinePeriodic(H_periodic, Sright_periodic)
for m in range(H_periodic.k + H_periodic.l * 2 - 1):
    state_periodic.evolve()
assert state_periodic.m == H_periodic.k + H_periodic.l * 2 - 1
states_hash = None
states = [state.shift(-H_periodic.l, clear_history=True) for state in state_periodic.state_dict.values()]
while True:
    states = periodic_evolve_states(states)
    new_states_hash = tuple(sorted(map(hash, states)))
    print(len(new_states_hash))
    if new_states_hash == states_hash:
        break
    else:
        states_hash = new_states_hash
unique_states = states.copy()
delta_Es_dict = {}
results = []
for unique_state in unique_states:
    new_state = unique_state.copy()
    states_all = periodic_evolve_states([new_state.copy()], c=c)
    states_all = {hash(state): state for state in states_all}
    if hash(new_state) not in states_all:
        continue
    Es = new_state.stab.energy_level.Es
    shape = Es.shape
    indices = np.indices(shape).reshape((len(shape), 2**len(shape))).T
    delta_Es = np.full(shape, max_float)
    for index in indices:
        Es = np.full(shape, max_float)
        Es[tuple(index)] = 0.0
        new_state.stab.energy_level.Es = Es
        states_all = periodic_evolve_states([new_state], clear_history=False, c=c)
        states_all = {hash(state): state for state in states_all}
        if hash(new_state) in states_all:
            final_state = states_all[hash(new_state)]
            energy_level = final_state.stab.energy_level
            result = final_state.stab.energy_level.get_state(tuple(index), H_periodic.original_paulis, return_stab=False)
            results.append(result)

for stab, energy in results:
    print('energy', energy)
    print(stab)
    print()
