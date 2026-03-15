from state import Hamiltonian, StateMachine, generate_Sright_all, FullStateEnergies


def get_ref_energy(stab, H):
    return sum([stab.get_energy(p) * c for p, c in H.original_paulis])


nexts = 10  # number of excited states

H = Hamiltonian.from_file("../example/hamiltonian_finite.txt")
Sright_all = generate_Sright_all(H)
lists = []

SM2 = StateMachine(H, Sright_all, nexts=nexts)
while True:
    SM2.evolve()
    #print(SM2.m, len(SM2.state_dict))
    lists.append(list(SM2.state_dict.values()))
    if SM2.m == H.n:
        break
result = SM2.generate_gs()
for stab, energy in result:
    print(stab, energy)
    assert abs(get_ref_energy(stab, H) - energy) < 1e-8

# full = FullStateEnergies(H.original_paulis)
# full.evolve()
# result_ref = full.generate_gs(nexts)
# for stab, energy in result_ref:
#     print(stab, energy)
