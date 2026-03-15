from LCSS_new import *
import matplotlib.pyplot as plt 
import time
from joblib import Parallel, delayed
import multiprocessing
np.random.seed(42)

H = Pauli_Hamiltonian.TI_local_H(["XX","Z"],20)
N  = 500
instance = LCSS(H)

print("LCSS States")
print("--------------------------")
t = time.time()
states  = instance.get_states(N)
triples = [triple.get_triple(s) for s in states]
t1 = time.time()
print(f"State Generation Time: {t1-t}")

N_states = len(states)
t = time.time()
def compute_overlap_row(i, triples, N_states):
    row = np.zeros(N_states, dtype=np.complex128)
    ti = triples[i]
    for j in range(i+1, N_states):
        row[j] = triple.inner(ti, triples[j])
    return i, row

n_jobs = multiprocessing.cpu_count()

results = Parallel(
    n_jobs=n_jobs,
    backend="loky",
    batch_size=1
)(
    delayed(compute_overlap_row)(i, triples, N_states)
    for i in range(N_states)
)

O = np.eye(N_states, dtype=np.complex128)

for i, row in results:
    O[i, i+1:] = row[i+1:]
    O[i+1:, i] = row[i+1:].conjugate()
t1 = time.time()
print(f"Overlap Generation Time: {t1-t}")

t = time.time()
new_states= [[t.apply_pauli(p) for p in instance.H.paulis] for t in triples]
def compute_energy_row(i, triples, new_states, states, H, N_states):
    row = np.zeros(N_states, dtype=np.complex128)

    # Diagonal term
    row[i] = np.dot(
        H.coeffs,
        [pauli_expect(p, states[i]) for p in H.paulis]
    )

    # Off-diagonal
    ti = triples[i]
    for j in range(i+1, N_states):
        val = np.dot(
            H.coeffs,
            [c * triple.inner(ti, t) for (c, t) in new_states[j]]
        )
        row[j] = val

    return i, row

results_M = Parallel(
    n_jobs=n_jobs,
    backend="loky",
    batch_size=1
)(
    delayed(compute_energy_row)(
        i, triples, new_states, states, instance.H, N_states
    )
    for i in range(N_states)
)

M = np.zeros((N_states, N_states), dtype=np.complex128)

for i, row in results_M:
    M[i, i:] = row[i:]
    M[i:, i] = row[i:].conjugate()
t1 = time.time()
print(f"Energy Generation Time: {t1-t}")
print("--------------------------")

print("Classical States")
print("--------------------------")
t = time.time()
states_Z  = instance.get_Z_states(N)  
t1 = time.time()
print(f"State Generation Time: {t1-t}")

t = time.time()
O_Z = np.eye(len(states_Z),dtype=complex)
for i,s1 in enumerate(states_Z):
    for j,s2 in enumerate(states_Z[i+1:]):
        if s1 == s2:
            O_Z[i,j+i+1] = 1
            O_Z[j+i+1,i] = 1
t1 = time.time()
print(f"State Generation Time: {t1-t}")

t = time.time()
M_Z=full_matrix(instance.H, states_Z)
t1 = time.time()
print(f"State Generation Time: {t1-t}")
print("--------------------------")

print("Post Proccessing")
rank_list = [np.linalg.matrix_rank(O[:i,:i], hermitian=True) for i in range(1,len(O))]
energies = [LCSS.basis_diagonalize(M[:i,:i],O[:i,:i])[0] for i in range(1,len(O))]
increases =  [0]+[i for i in range(1, len(rank_list)) if rank_list[i] > rank_list[i - 1]] 
energies = np.array(energies)[increases]

rank_list_Z = [np.linalg.matrix_rank(O_Z[:i,:i], hermitian=True) for i in range(1,len(O_Z))]
energies_Z = [LCSS.basis_diagonalize(M_Z[:i,:i],O_Z[:i,:i])[0] for i in range(1,len(O_Z))]
increases_Z =  [0]+[i for i in range(1, len(rank_list_Z)) if rank_list_Z[i] > rank_list_Z[i - 1]] 
energies_Z = np.array(energies_Z)[increases_Z]

g = run_dmrg(convert_to_mpos(H.paulis,H.coeffs),50)

plt.plot(np.abs((energies-g)/g),label='Stabilizer')
plt.plot(np.abs((energies_Z-g)/g),label='Computational')
plt.savefig("Test_Plot.pdf",dpi=200)
plt.legend()
plt.grid(alpha=0.2)
print("Plot Saved")
