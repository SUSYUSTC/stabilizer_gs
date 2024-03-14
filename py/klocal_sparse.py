import sys
from state import Hamiltonian, StateMachine, generate_Sright_all

H = Hamiltonian.from_file("../hamiltonian_finite.txt")
Sright_all = generate_Sright_all(H)
lists = []
SM2 = StateMachine(H, Sright_all)
while True:
    SM2.evolve()
    print(SM2.m, len(SM2.state_dict))
    lists.append(list(SM2.state_dict.values()))
    if SM2.m == H.n:
        break
SM2.generate_gs()
energy_ref = sum([SM2.stab_gs.get_energy(P) * w for P, w in H.original_paulis])

print(SM2.energy_gs)
print(energy_ref)
print(SM2.stab_gs)
