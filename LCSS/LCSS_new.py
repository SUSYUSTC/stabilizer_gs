import sys 
sys.path.append("../")
sys.path.append("../py")
from py.state import Hamiltonian, StateMachine, generate_Sright_all
from py.DMRG import *
from py.pauli import Pauli
import numpy as np
from functools import reduce
from scipy.linalg import eigh

import numpy as np



def find_nonzero_row(A, pivot_row, col):
    rows = np.flatnonzero(A[pivot_row:, col])
    return rows[0] + pivot_row if rows.size else -1

def upper(A):
    """Force upper-triangular structure mod 2"""
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            if i < j:
                A[i, j] = (A[i, j] ^ A[j, i]) & 1
            elif i > j:
                A[i, j] = 0
            else:
                A[i, j] &= 1

# ============================================================
# Bit-packed GF(2) linear algebra
# Replaces: rref, gf2_solve, gf2_nullspace
# ============================================================

def _pack_rows(A):
    """Pack uint8 matrix rows into Python ints"""
    rows = []
    for i in range(A.shape[0]):
        r = 0
        row = A[i]
        for j in range(A.shape[1]):
            if row[j]:
                r |= (1 << j)
        rows.append(r)
    return rows, A.shape[1]


def _unpack_rows(rows, n_cols):
    A = np.zeros((len(rows), n_cols), dtype=np.uint8)
    for i, r in enumerate(rows):
        for j in range(n_cols):
            A[i, j] = (r >> j) & 1
    return A


def rref(A):
    """
    Row-reduced echelon form over GF(2)
    Drop-in replacement for your original rref
    """
    rows, n_cols = _pack_rows(A)
    n_rows = len(rows)

    pivot_row = 0
    pivots = []

    for col in range(n_cols):
        # find pivot
        pr = -1
        for r in range(pivot_row, n_rows):
            if (rows[r] >> col) & 1:
                pr = r
                break
        if pr == -1:
            continue

        # swap
        rows[pivot_row], rows[pr] = rows[pr], rows[pivot_row]

        # eliminate
        pivot_val = rows[pivot_row]
        for r in range(n_rows):
            if r != pivot_row and ((rows[r] >> col) & 1):
                rows[r] ^= pivot_val

        pivots.append(col)
        pivot_row += 1
        if pivot_row == n_rows:
            break

    R = _unpack_rows(rows, n_cols)
    return R, pivots, pivot_row


def gf2_solve(A, b):
    """
    Solve A x = b over GF(2)
    Drop-in replacement
    """
    b = b.reshape(-1, 1)
    M = np.hstack((A, b))
    R, pivots, rank = rref(M)

    n = A.shape[1]

    # consistency check
    for i in range(R.shape[0]):
        if R[i, n] and not np.any(R[i, :n]):
            return None

    x = np.zeros(n, dtype=np.uint8)
    for i, p in enumerate(pivots):
        x[p] = R[i, n]

    return x


def gf2_nullspace(A):
    """
    Nullspace of A over GF(2)
    Drop-in replacement
    """
    R, pivots, rank = rref(A)
    n = A.shape[1]

    pivot_set = set(pivots)
    free = [j for j in range(n) if j not in pivot_set]

    if not free:
        return np.zeros((n, 0), dtype=np.uint8)

    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(pivots):
            v[p] = R[i, f]
        basis.append(v)

    return np.array(basis, dtype=np.uint8).T


def gf2_block_reduce(Q):
    M = Q.copy().astype(np.uint8)
    n = M.shape[0]

    P = np.eye(n, dtype=np.uint8)
    active = np.ones(n, dtype=bool)
    spaces = []

    idx = np.arange(n)

    while active.any():
        i = idx[active][0]

        row_i = M[i]
        candidates = idx[(row_i == 1) & active & (idx != i)]

        if candidates.size:
            j = candidates[0]

            ks = idx[active & (idx != i) & (idx != j)]

            Mi = M[i]
            Mj = M[j]

            alpha = Mj[ks]   # M[j,k]
            beta  = Mi[ks]   # M[i,k]

            # rows
            M[ks] ^= alpha[:, None] * Mi
            M[ks] ^= beta[:, None]  * Mj

            # columns
            M[:, ks] ^= M[:, i][:, None] * alpha
            M[:, ks] ^= M[:, j][:, None] * beta

            # transformation
            P[ks] ^= alpha[:, None] * P[i]
            P[ks] ^= beta[:, None]  * P[j]

            upper(M)   
            active[i] = False
            active[j] = False
            spaces.append([i, j])

        else:
            active[i] = False
            spaces.append([i])

    return M, P, spaces



def get_intersection(V1, V2, b1, b2):
    """
    Intersection of two affine subspaces over GF(2)

    Returns:
        (particular solution, nullspace basis) or None
    """
    b = (b1 ^ b2) & 1

    if V1.shape[1] == 0:
        A = V2
    elif V2.shape[1] == 0:
        A = V1
    else:
        A = np.hstack([V1, V2])

    sol = gf2_solve(A, b)
    if sol is None:
        return None
    return sol, gf2_nullspace(A)

# -------------------------------------------------------------------
# Pauli lookup tables
# -------------------------------------------------------------------

PAULI_MAP = np.zeros((4, 2), dtype=np.uint8)
PAULI_MAP[0] = [0, 0]   # '_'
PAULI_MAP[1] = [1, 0]   # 'X'
PAULI_MAP[2] = [1, 1]   # 'Y'
PAULI_MAP[3] = [0, 1]   # 'Z'

PAULI_INV = {
    (0, 0): '_',
    (1, 0): 'X',
    (1, 1): 'Y',
    (0, 1): 'Z',
}


def pauli_index(c):
    if c == 'X':
        return 1
    if c == 'Y':
        return 2
    if c == 'Z':
        return 3
    return 0   # '_'


# -------------------------------------------------------------------
# Pauli <-> binary
# -------------------------------------------------------------------

def pauli_to_bin(s):
    """
    Convert Pauli string to binary tableau row.
    """
    n = len(s) - 1
    out = np.zeros(2 * n + 1, dtype=np.uint8)

    for i in range(n):
        idx = pauli_index(s[i + 1])
        out[i] = PAULI_MAP[idx][0]
        out[i + n] = PAULI_MAP[idx][1]

    out[2 * n] = 0 if s[0] == '+' else 1
    return out


def bin_to_pauli(arr):
    """
    Convert binary tableau row back to Pauli string.
    """
    n = (arr.shape[0] - 1) // 2
    chars = []

    for i in range(n):
        chars.append(PAULI_INV[(int(arr[i]), int(arr[i + n]))])

    return ('+' if arr[-1] == 0 else '-') + ''.join(chars)


# -------------------------------------------------------------------
# Group / symplectic utilities
# -------------------------------------------------------------------
def symplectic_complement(l):
    V = np.array(
        [pauli_to_bin(p)[:-1] for p in l],
        dtype=np.uint8
    )

    n = V.shape[1] // 2

    P = np.block([
        [np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
        [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)]
    ])

    W = gf2_nullspace(V @ P)

    A = np.hstack([V.T, W])
    A_red = rref(A)[0][:, V.shape[0]:]

    pivots = []
    pivot_row = V.shape[0]

    for col in range(A_red.shape[1]):
        row = find_nonzero_row(A_red, pivot_row, col)
        if row >= 0:
            pivots.append(col)
            pivot_row = row + 1

    return W[:, pivots], V.T

def full_extend_group(group):
    n  = len(group[0])-1
    k = n - len(group)
    W,_ = symplectic_complement(group)
    
    new_ops = W[:,:k]
    groups = []
    for x in range(1 << k):       
        bits = np.array([(x >> i) & 1 for i in range(k)], dtype=np.uint8)
        groups.append(group + [bin_to_pauli(np.concatenate([n,[b]])) for (n,b) in zip(new_ops.T,bits)])
    return groups


# -------------------------------------------------------------------
# Phase-aware Gaussian elimination
# -------------------------------------------------------------------

def phase_add_rows(v1, v2):
    L = v1.shape[0]
    n = (L - 1) // 2

    t0 = 0   # q2 @ p1
    t1 = 0   # q1 @ p1 + q2 @ p2 + 3 * (...)

    # t0 = q2 @ p1
    for i in range(n):
        t0 += int(v2[i]) * int(v1[n + i])

    # q1 @ p1 + q2 @ p2
    for i in range(n):
        t1 += int(v1[i]) * int(v1[n + i])
        t1 += int(v2[i]) * int(v2[n + i])

    # 3 * ((q1 + q2)%2 @ (p1 + p2)%2)
    for i in range(n):
        t1 += 3 * ((v1[i] + v2[i]) & 1) * ((v1[n + i] + v2[n + i]) & 1)

    c = t0 + (t1 // 2)

    # row addition mod 2
    for i in range(L):
        v2[i] ^= v1[i]

    # phase update
    v2[L - 1] ^= (c & 1)


def phase_elim_rows(A, pivot_row, col, tracker):
    for row in range(A.shape[0]):
        if A[row, col] and row != pivot_row:
            phase_add_rows(A[pivot_row], A[row])
            tracker[row] ^= tracker[pivot_row]


def phase_rref(A_init):
    A = np.copy(A_init)
    tracker = np.eye(A.shape[0], dtype=np.uint8)

    pivot_row = 0

    for col in range(A.shape[1]):
        row = find_nonzero_row(A, pivot_row, col)
        if row >= 0:
            A[[pivot_row, row]] = A[[row, pivot_row]]
            tracker[[pivot_row, row]] = tracker[[row, pivot_row]]
            phase_elim_rows(A, pivot_row, col, tracker)
            pivot_row += 1

    return A, tracker, pivot_row


def canonical(group):
    n = len(group)
    A = np.array(
        [pauli_to_bin(p) for p in group],
        dtype=np.uint8
    )

    A, _, _ = phase_rref(A)

    for i in range(A.shape[0]):
        A[i, -1] ^= ((A[i, :n] @ A[i, n:2*n]) % 4) // 2

    return A & 1

def pauli_expect(p, group):
    A = np.array([pauli_to_bin(g)[:-1] for g in group])
    sol = gf2_solve(A.T, pauli_to_bin("+"+p)[:-1])

    if sol is None:
        return 0

    if sol is not None:
        A = np.array([pauli_to_bin(g) for g in group])
        v = np.zeros(2 * len(p)+1, dtype=np.uint8)
        for i,val in enumerate(sol):
            if val:
                phase_add_rows(A[i], v)
        return (-1+0j)**(v[-1])

class triple:
    def __init__(self, l, Q, V, z0):
        self.l = np.asarray(l, dtype=np.uint8)
        self.Q = np.asarray(Q, dtype=np.uint8)
        self.V = np.asarray(V, dtype=np.uint8)
        self.z0 = np.asarray(z0, dtype=np.uint8)

        self.n = self.z0.shape[0]
        self.k = self.Q.shape[0]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def get_triple(group):
        A = canonical(group)
        n = A.shape[0]

        q = []
        p = []
        rho = []
        c = []
        gamma = []

        for i in range(n):
            if np.any(A[i, :n] != 0):
                q.append(A[i, :n])
                p.append(A[i, n:2*n])
                c.append(A[i, -1])
            else:
                rho.append(A[i, n:2*n])
                gamma.append(A[i, -1])

        q = np.array(q, dtype=np.uint8)
        if len(q) ==0:
            q = np.zeros((0,n),dtype=np.uint8)
        p = np.array(p, dtype=np.uint8)
        rho = np.array(rho, dtype=np.uint8)
        gamma = np.array(gamma, dtype=np.uint8)

        z0 = np.zeros(n, dtype=np.uint8)
        if gamma.shape[0] > 0:
            z0 = gf2_solve(rho, gamma) % 2

        k = q.shape[0]
        l = np.array([(x @ y) for (x, y) in zip(p, q)], dtype=np.uint8) % 2

        Q = np.zeros((k, k), dtype=np.uint8)

        for i in range(k):
            Q[i, i] = (c[i] + (p[i] @ z0)) % 2
            for j in range(i + 1, k):
                Q[i, j] = (
                    (p[j] @ q[i])
                    + (p[i] @ q[i]) * (p[j] @ q[j])
                ) % 2

        Q &= 1
        return triple(l, Q, q, z0)

    # ------------------------------------------------------------------
    # Inner product machinery
    # ------------------------------------------------------------------

    @staticmethod
    def __inner_terms__(triple1, triple2):
        overlap = get_intersection(
            triple1.V.T,
            triple2.V.T,
            triple1.z0,
            triple2.z0
        )

        if overlap is None:
            return (
                0,
                np.zeros(0, dtype=np.uint8),
                np.zeros((0, 0), dtype=np.uint8),
            )

        b, M = overlap

        N = (
            1 / np.sqrt(2 ** triple1.k)
            * 1 / np.sqrt(2 ** triple2.k)
        )

        phase = (
            (-1 + 0j) ** (b[:triple1.k] @ triple1.Q @ b[:triple1.k])
            * (-1 + 0j) ** (b[triple1.k:] @ triple2.Q @ b[triple1.k:])
            * (-1j) ** ((triple1.l @ b[:triple1.k]) % 2)
            * (1j) ** ((triple2.l @ b[triple1.k:]) % 2)
        )

        if M.shape[1] == 0:
            return (
                phase * N,
                np.zeros(0, dtype=np.uint8),
                np.zeros((0, 0), dtype=np.uint8),
            )

        P1 = M[:triple1.k]
        P2 = M[triple1.k:]

        l1_new = P1.T @ triple1.l
        l2_new = P2.T @ triple2.l
        l_new = (l1_new + l2_new) % 2

        Q_new = (
            P1.T @ triple1.Q @ P1
            + P2.T @ triple2.Q @ P2
            + np.outer(l1_new, l2_new)
        ) % 2

        Ql = (
            P1.T @ (triple1.Q + triple1.Q.T) @ b[:triple1.k]
            + P2.T @ (triple2.Q + triple2.Q.T) @ b[triple1.k:]
            + P1.T @ np.outer(triple1.l, triple1.l) @ b[:triple1.k]
            + P2.T @ np.outer(triple2.l, triple2.l) @ b[triple1.k:]
            + l1_new
        ) % 2

        Ql = (Ql + np.diag(Q_new)) % 2
        np.fill_diagonal(Q_new, Ql)

        return phase * N, l_new, Q_new

    # ------------------------------------------------------------------
    # Block sums
    # ------------------------------------------------------------------

    @staticmethod
    def __block_sum__(Qb, blocks):
        prod = 1
        for b in blocks:
            if len(b) == 1:
                if Qb[b[0], b[0]] == 0:
                    prod *= 2
                else:
                    return 0
            else:
                p = (
                    1
                    + (-1 + 0j) ** Qb[b[0], b[0]]
                    + (-1 + 0j) ** Qb[b[1], b[1]]
                    + (-1 + 0j) ** (
                        Qb[b[0], b[0]]
                        + Qb[b[0], b[1]]
                        + Qb[b[1], b[0]]
                        + Qb[b[1], b[1]]
                    )
                )
                if p == 0:
                    return 0
                prod *= p
        return prod

    @staticmethod
    def __twisted_block_sum__(Q, l):
        Qb, B, blocks = gf2_block_reduce(Q)
        lb = B @ l
        Qlb = (Qb + np.diag(lb)) % 2

        quad_term = triple.__block_sum__(Qb, blocks)
        lin_term = triple.__block_sum__(Qlb, blocks)

        return (
            (quad_term + lin_term) / 2
            + 1j * (quad_term - lin_term) / 2
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def inner(triple1, triple2):
        p, l, Q = triple.__inner_terms__(triple1, triple2)
        upper(Q)
        return p * triple.__twisted_block_sum__(Q, l)

    def apply_pauli(self, P):
        v = pauli_to_bin("+" + P)
        q = v[:len(P)]
        p = v[len(P):2 * len(P)]

        Q_new = self.Q + np.diag(self.V @ p)
        z0_new = self.z0 + q

        phase = (1j) ** (q @ p) * (-1 + 0j) ** (p @ self.z0)

        return phase, triple(self.l, Q_new, self.V, z0_new)



### Linear Combination of Pauli Strings Hamiltonians  
class Pauli_Hamiltonian():
    
    def __init__(self,coeffs,paulis):
        """
        Initialize

        Args:
            coeffs (np.array[float]): coefficients
            paulis (list[string]): paulis
        """
        self.coeffs = coeffs
        self.paulis = paulis
    
    def convert_to_spare_hamiltonian(self):
        strings = []
        indicies = []
        for p in self.paulis:
            s = ""
            idx = []
            for (i,v) in enumerate(p):
                if(v != '_'):
                    idx.append(i)
                    s+=v 
            strings.append(s)
            indicies.append(idx)
        n = len(self.paulis[0])
        diffs = [np.max(i)-np.min(i)+1 for i in indicies]
        k = np.max(diffs)
        
        H  = Hamiltonian(n,k)
        for i in range(len(strings)):
            H.add_Pauli(Pauli.from_string(strings[i],indicies[i]),self.coeffs[i])
        
        return H 
    
    def __get_ref_energy__(stab, H):
        return sum([stab.get_energy(p) * c for p, c in H.original_paulis])

    def __stab_to_str__(stab,n_qubits):
        s1 = (stab.begin - 0) * "_"
        s2 = (n_qubits - stab.end) * "_"
        return [s1+"".join(i)+s2 for i in np.array(['_', 'X', 'Z', 'Y'])[stab.Ps.array4]]

    def get_excited_states(self,k):
        sparseH = self.convert_to_spare_hamiltonian()
        Sright_all = generate_Sright_all(sparseH)
        lists = []

        SM2 = StateMachine(sparseH, Sright_all, nexts=k)
        while True:
            SM2.evolve()
            lists.append(list(SM2.state_dict.values()))
            if SM2.m == sparseH.n:
                break
        result = SM2.generate_gs()
        result = [(stab,energy) for (stab,energy) in result if stab is not None]
        for stab, energy in result:
            assert abs(Pauli_Hamiltonian.__get_ref_energy__(stab, sparseH) - energy) < 1e-8
        
        groups = [Pauli_Hamiltonian.__stab_to_str__(r[0],sparseH.n) for r in result]
        phases = [i[0].Ps.phase   for i in result]
        energies = [r[1] for r in result]
        
            
        return [['+'+x if (1+_) else '-'+x for (x,_) in zip(g,p)] for (g,p) in zip(groups, phases)], energies 

    @staticmethod
    def TI_local_op(op, n):
        k = len(op)
        paulis = ["".join([op if _==i else "_" for _ in range(n-k+1)])for i in range(n-k+1)]
        return paulis
    @staticmethod
    def TI_local_H(op_list, n):
        paulis = []
        for op in op_list:
            paulis += Pauli_Hamiltonian.TI_local_op(op,n)
        return Pauli_Hamiltonian(np.random.random(len(paulis))*2-1, paulis)

    def Z_component(self):
        coeffs = []
        paulis = []
        for (c,p) in zip(self.coeffs, self.paulis):
            if not ("X" in p or "Y" in p):
                coeffs.append(c)
                paulis.append(p)
        return Pauli_Hamiltonian(coeffs, paulis)

### Full LCSS 
class LCSS:
    def __init__(self, H):
        self.H = H 
    
    def get_states(self, N):
        states, energies = self.H.get_excited_states(N)
        extended_states = []
        for s in (states):
            extended = full_extend_group(s)
            for e in extended:
                extended_states.append(e)
        return extended_states 
    
    
    def get_Z_states(self, N):
        states, energies = self.H.Z_component().get_excited_states(N)
        extended_states = []
        for s in (states):
            extended = extend_Z_group(s)
            for e in extended:
                extended_states.append(get_Z_state(e))
        return extended_states 
    
    def run_LCSS(self, N):
        print("------------extension--------------------")
        states  = self.get_states(N)
        triples = [triple.get_triple(s) for s in states]
        N_states = len(states)
        print("------------overlap--------------------")
        O = np.eye(N_states,dtype=complex)
        for i in range(0,N_states):
            for j in range(i+1,N_states):
                O[i,j] = triple.inner(triples[i], triples[j])
                O[j,i] = O[i,j].conjugate()
                
        new_states= [[t.apply_pauli(p) for p in self.H.paulis] for t in triples]
        print("------------energies--------------------")
        M = np.eye(N_states,dtype=complex)
        for i in range(N_states):
            M[i,i] = np.dot(self.H.coeffs,[pauli_expect(p, states[i]) for p in self.H.paulis])
            for j in range(i+1,N_states):
                M[i,j] = np.dot(self.H.coeffs,[c*triple.inner(triples[i],t) for (c,t) in (new_states[j])])
                M[j,i] = M[i,j].conjugate() 
        return M, O 
    
    def run_Z_LCSS(self, N):
        states  = self.get_Z_states(N)  
        O = np.eye(len(states),dtype=complex)
        for i,s1 in enumerate(states):
            for j,s2 in enumerate(states[i+1:]):
                if s1 == s2:
                    O[i,j+i+1] = 1
                    O[j+i+1,i] = 1
        return full_matrix(self.H, states), O
    
    @staticmethod
    def basis_diagonalize(M,O):
        O_data = eigh(O)
        valid = [i for i,v in enumerate(O_data[0]) if not np.isclose(v,0)]

        U_plus = O_data[1][:, valid]
        data = eigh(U_plus.conjugate().T @ M @ U_plus, np.diag(O_data[0][valid]))

        E = data[0][0]
        v = U_plus @ data[1][:,0]
        return E,v
    
def pack_paulis(H):
    arr = np.array([pauli_to_bin('+'+p)[:-1] for p in H.paulis])
    N, total = arr.shape
    n = (total) // 2

    # Split
    bits1 = arr[:, :n]
    bits2 = arr[:, n:2*n]

    # Precompute bit weights (MSB first)
    weights = (1 << np.arange(n-1, -1, -1, dtype=np.uint64))

    # Convert bits to integers
    A = bits1 @ weights
    B = bits2 @ weights
    return A, B 

def get_matrix(a,b, states):
    N = len(states)
    M = np.zeros((N,N), dtype=complex)
    
    for i1, s1 in enumerate(states):
        new_state = s1 ^ a
        coeff = (1j)**(bin(a & b).count('1')) * (-1)**(bin(b & s1).count('1'))
        for i2, s2 in enumerate(states):
            if s2 == new_state:
                M[i2, i1] = coeff 
    return M

def full_matrix(H, states):
    A, B = pack_paulis(H)
    N = len(states)
    out =  np.zeros((N,N), dtype=complex)
    for i, (a,b) in enumerate(zip(A, B)):
        out += H.coeffs[i] * get_matrix(a,b,states)
    return out 

def extend_Z_group(s):
    M = np.array([[1 if v!="_" else 0 for v in p[1:]] for p in s], dtype=np.uint8).T
    n, k = M.shape               

    A = rref(np.hstack([M, np.eye(n, dtype=np.uint8)]))[0][:,k:]
    pivots = []
    pivot_row = k

    for col in range(A.shape[1]):
        row = find_nonzero_row(A, pivot_row, col)
        if row >= 0:
            pivots.append(col)
            pivot_row = row + 1
    new_ops = np.eye(n, dtype=np.uint8)[:, pivots]

    new_ops = ["".join(["Z" if v else "_" for v in p]) for p in new_ops.T]
    m = len(new_ops)
    groups = []
    for x in range(1 << m):       
        bits = np.array([(x >> i) & 1 for i in range(m)], dtype=np.uint8)
        groups += [s + ['-'+new_ops[i] if b else '+'+new_ops[i] for i,b in enumerate(bits)]]
    return groups

def get_Z_state(s):
    n = len(s)
    M = np.array([pauli_to_bin(p) for  p in s])
    A, b = M[:,n:2*n], M[:,-1]
    weights = (1 << np.arange(n-1, -1, -1, dtype=np.uint64))
    return weights @ gf2_solve(A,b)