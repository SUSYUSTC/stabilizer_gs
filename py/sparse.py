import sys
from stab import StabilizerGroup
from state import Hamiltonian, FullStateEnergies

H = Hamiltonian.from_file(sys.argv[1])
print(*H.original_paulis, sep='\n')

# use sparse (and nonlocal) algorithm
fullstateE = FullStateEnergies(H.original_paulis)
#fullstateE.evolve()
full_Ps, full_energy_gs = fullstateE.evolve()
full_stab = StabilizerGroup(full_Ps)
print('gs energy', full_energy_gs)
# stabilizer group S
print(full_stab.Ps)
