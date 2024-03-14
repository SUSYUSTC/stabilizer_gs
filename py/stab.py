import typing as tp
import numpy as np
import linear
from utils import inttype, zerosint2, randomint2
from pauli import Pauli


class StabilizerGroup:
    def __init__(self, Ps: Pauli, canonicalized=False):
        '''
        create empty stabilizer group S
        '''
        self.Ps = Ps
        self.initialize()
        if not canonicalized:
            self.canonicalize()

    @property
    def begin(self):
        return self.Ps.begin

    @property
    def end(self):
        return self.Ps.end

    def initialize(self):
        self.m, self.dim = self.Ps.array.shape
        self.nqubits = self.dim // 2

    def flip_sign(self, index):
        self.Ps.ZX_phase[index] = (self.Ps.ZX_phase[index] + 2) % 4

    def transform(self, U):
        '''
        P'i -> prod_j Pj^{U_ij}, U_ij = 0, 1
        '''
        Ps = self.Ps.copy()
        Ps.x = U.dot(self.Ps.x) % 2
        Ps.z = U.dot(self.Ps.z) % 2
        Ps.group_phase = 0
        signs = np.array([self.decompose(P)[1] for P in Ps], dtype=inttype)
        Ps = Ps * signs
        self.Ps = Ps

    def canonicalize(self):
        '''
        unique representation of the stabilizer group so that can be hashed
        '''
        _, U = linear.Gauss_elimination_full(self.Ps.array, 2, allow_reorder=True)
        self.transform(U)

    def __repr__(self):
        return repr(self.Ps)

    def hash(self, add_position=True):
        return self.Ps.hash(add_position=add_position)

    def __hash__(self):
        return self.hash()

    @classmethod
    def empty(cls, nqubits, begin=0):
        z = np.zeros((0, nqubits), dtype=inttype)
        x = np.zeros((0, nqubits), dtype=inttype)
        Ps = Pauli(z, x, begin=begin)
        return StabilizerGroup(Ps)

    @classmethod
    def random_stab(cls, nqubits, m):
        assert nqubits >= m
        array = zerosint2((0, nqubits * 2))
        Ps = Pauli.from_array(array)
        stab = StabilizerGroup(Ps)
        for _ in range(m):
            # xz + zx = 0: commutation
            full = linear.Gauss_elimination_Aonly(Pauli(Ps.x, Ps.z).array, 2, allow_reorder=True)
            kernel = linear.get_kernel_reordered(full, 2).T
            while True:
                x = np.random.random(len(kernel)) > 0.5
                phase = 1 - randomint2(()) * 2
                P = Pauli.from_array(x.dot(kernel), phase=phase)
                if np.any(x > 0) and (not stab.include(P)):
                    break
                else:
                    continue
            Ps = Pauli.concatenate((Ps, P[None]))
            array = Ps.array
            stab = StabilizerGroup(Ps)
        return stab

    def range(self, begin, end, allow_truncation=False):
        if not allow_truncation:
            assert begin <= self.Ps.begin
            assert end >= self.Ps.end
        Ps_new_range = self.Ps.range(begin, end, allow_truncation=allow_truncation)
        return StabilizerGroup(Ps_new_range, canonicalized=(not allow_truncation))

    def copy(self):
        return StabilizerGroup(self.Ps.copy(), canonicalized=True)

    def shift(self, add):
        return StabilizerGroup(self.Ps.shift(add), canonicalized=True)

    def add(self, P: Pauli):
        self = self.copy()
        self.Ps = Pauli.concatenate((self.Ps, P[None]))
        self.canonicalize()
        return self

    def decompose(self, P: Pauli):
        '''
        P = \pm \prod_i S_i
        '''
        Ps, P = Pauli.extend(self.Ps, P)
        is_over, is_under, x = linear.solve_modN(Ps.array.T, P.array, 2)
        if is_over:
            return None
        if is_under:
            raise ValueError('Underdetermined')
        product = Ps[x > 0].reduce_dot()
        assert np.all(product.array == P.array)
        return x, P.phase * product.phase
        '''
        try:
            x = linear.solve_modN(Ps.array.T, P.array, 2) > 0
            product = Ps[x].reduce_dot()
            assert np.all(product.array == P.array)
            return x, P.phase * product.phase
        except linear.OverDetermined:
            return None
        '''

    def include(self, P: Pauli) -> bool:
        '''
        whether \pm P \in S
        '''
        return self.decompose(P) is not None

    def commute(self, P: Pauli) -> bool:
        '''
        All elements commute with pauli
        '''
        return np.all(self.Ps.commute(P))

    def get_energy(self, P: Pauli):
        result = self.decompose(P)
        if result is None:
            return 0.0
        else:
            x, phase = result
            return phase

    def project(self, begin, end):
        assert begin >= self.Ps.begin
        assert end <= self.Ps.end
        Ps = self.Ps
        n_left = begin - Ps.begin
        n_middle = end - begin
        # n_right = Ps.end - end
        assert len(Ps.batch_size) == 1

        x = Ps.x
        z = Ps.z
        x_out = np.concatenate([x[:, :n_left], x[:, n_left + n_middle:]], axis=-1)
        z_out = np.concatenate([z[:, :n_left], z[:, n_left + n_middle:]], axis=-1)
        array_out = Pauli(z_out, x_out).array.T

        kernel = linear.get_kernel(array_out, 2)
        if kernel.shape[1] == 0:
            #return StabilizerGroup(Pauli.identity(n_middle, size=(0, ), begin=begin))
            return StabilizerGroup.empty(n_middle, begin=begin)
        new_Ps = Pauli.stack([Ps[include > 0].reduce_dot() for include in kernel.T])
        return StabilizerGroup(new_Ps.range(begin, end, allow_truncation=True))


class EnergyLevel:
    def __init__(self, Es, nPs, Ps_indices, Ps_signs):
        self.Es = Es
        self.nPs = nPs
        self.Ps_indices = Ps_indices
        self.Ps_signs = Ps_signs

    @property
    def m(self):
        return self.Es.ndim

    @classmethod
    def empty(cls, nmax):
        Es = np.zeros(())
        nPs = np.zeros((), dtype=inttype)
        Ps_indices = np.zeros((nmax, ), dtype=inttype) - 1
        Ps_signs = np.zeros((nmax, ), dtype=inttype)
        return EnergyLevel(Es, nPs, Ps_indices, Ps_signs)

    def flip_sign(self, index):
        tmp_index = [slice(None) for _ in range(self.m)]
        tmp_index[index] = slice(None, None, -1)
        self.Es = self.Es[tuple(tmp_index)]
        self.nPs = self.nPs[tuple(tmp_index)]
        self.Ps_indices = self.Ps_indices[tuple(tmp_index)]
        self.Ps_signs = self.Ps_signs[tuple(tmp_index)]

    def copy(self):
        return EnergyLevel(self.Es.copy(), self.nPs.copy(), self.Ps_indices.copy(), self.Ps_signs.copy())

    def transform(self, U):
        indices = self.get_indices()
        indices_transform = U.dot(indices) % 2

        Es = np.zeros_like(self.Es)
        Es[tuple(indices_transform)] = self.Es[tuple(indices)]
        self.Es = Es

        nPs = np.zeros_like(self.nPs)
        nPs[tuple(indices_transform)] = self.nPs[tuple(indices)]
        self.nPs = nPs

        Ps_indices = np.zeros_like(self.Ps_indices)
        Ps_indices[tuple(indices_transform)] = self.Ps_indices[tuple(indices)]
        self.Ps_indices = Ps_indices

        Ps_signs = np.zeros_like(self.Ps_signs)
        Ps_signs[tuple(indices_transform)] = self.Ps_signs[tuple(indices)]
        self.Ps_signs = Ps_signs

    def add(self, index=-1):
        indices = np.indices((2, ) * self.m)
        self.Ps_indices[tuple(indices) + (self.nPs, )] = index
        Ps_signs1 = self.Ps_signs.copy()
        Ps_signs2 = self.Ps_signs.copy()
        Ps_signs1[tuple(indices) + (self.nPs, )] = 1
        Ps_signs2[tuple(indices) + (self.nPs, )] = -1
        self.nPs += 1

        self.Es = np.stack([self.Es, self.Es], axis=-1)
        self.nPs = np.stack([self.nPs, self.nPs], axis=-1)
        self.Ps_indices = np.stack([self.Ps_indices, self.Ps_indices], axis=-2)
        self.Ps_signs = np.stack([Ps_signs1, Ps_signs2], axis=-2)

    def project_to_next(self):
        level1 = EnergyLevel(self.Es[0], self.nPs[0], self.Ps_indices[0], self.Ps_signs[0])
        level2 = EnergyLevel(self.Es[1], self.nPs[1], self.Ps_indices[1], self.Ps_signs[1])
        return EnergyLevel.merge_gs(level1, level2)
        '''
        argmin = np.argmin(self.Es, axis=0)
        indices = np.indices((2, ) * (self.m - 1))
        index = (argmin, ) + tuple(indices)
        self.Es = self.Es[index]
        self.nPs = self.nPs[index]
        self.Ps_indices = self.Ps_indices[index]
        self.Ps_signs = self.Ps_signs[index]
        '''

    '''
    @classmethod
    def project_multiple_to_next(cls, states: tp.List["EnergyLevel"]) -> tp.List["EnergyLevel"]:
        #return functools.reduce(EnergyLevel.merge_gs, states)
    '''

    def get_indices(self):
        return np.indices((2, ) * self.m).reshape(self.m, 2**self.m)

    def add_energy(self, x, sign, w):
        indices = self.get_indices()
        signs = (1 - (x.dot(indices) % 2) * 2) * sign
        self.Es[tuple(indices)] += signs * w

    @classmethod
    def merge_gs(cls, state1: "EnergyLevel", state2: "EnergyLevel") -> "EnergyLevel":
        '''
        Use the lower energy as the energy of the merged state
        '''
        keep1 = state1.Es < state2.Es
        keep2 = ~keep1
        #Es_ref = np.min([state1.Es, state2.Es], axis=0)
        Es = state1.Es * keep1 + state2.Es * keep2
        nPs = state1.nPs * keep1 + state2.nPs * keep2
        Ps_indices = state1.Ps_indices * keep1[..., None] + state2.Ps_indices * keep2[..., None]
        Ps_signs = state1.Ps_signs * keep1[..., None] + state2.Ps_signs * keep2[..., None]
        return cls(Es, nPs, Ps_indices, Ps_signs)

    def clear(self):
        self.Es = np.full_like(self.Es, 0)
        self.nPs = np.full_like(self.nPs, 0)
        self.Ps_indices = np.full_like(self.Ps_indices, -1)
        self.Ps_signs = np.full_like(self.Ps_indices, 0)

    def get_state(self, index, paulis, return_stab=True):
        # find state & signs with the lowest energy
        energy = self.Es[index]
        nPs = self.nPs[index]
        Ps_index = self.Ps_indices[index][0:nPs]
        sign_index = self.Ps_signs[index][0:nPs]
        Ps = []
        for index, sign in zip(Ps_index, sign_index):
            P, _ = paulis[index]
            Ps.append(P * sign)
        if len(Ps) == 0:
            result = None
        else:
            if return_stab:
                result = StabilizerGroup(Pauli.stack(Ps))
            else:
                result = Pauli.stack(Ps)
        return (result, energy)


class StabilizerGroupEnergies(StabilizerGroup):
    def __init__(self, Ps: Pauli, energy_level: EnergyLevel, canonicalized=False):
        '''
        Here (2, 2, ..., 2) = (2, ) * k
        Es: (2, 2, ..., 2): minimial energy of each sign combination (+1, -1, ...)
        nPs, Ps_indices, Ps_signs represent the stabilizers of the full ground state (recording the history to get wavenumber)
        nPs: (2, 2, ..., 2)
        Ps_indices: (2, 2, ..., 2, n)
        Ps_signs: (2, 2, ..., 2, n)
        n represents at most n independent stabilizers
        '''
        self.Ps = Ps
        self.energy_level = energy_level
        self.initialize()
        if not canonicalized:
            self.canonicalize()
        assert np.all(self.Ps.phase == 1)

    @classmethod
    def empty(cls, nqubits, nmax, begin=0):
        z = np.zeros((0, nqubits), dtype=inttype)
        x = np.zeros((0, nqubits), dtype=inttype)
        Ps = Pauli(z, x, begin=begin)
        energy_level = EnergyLevel.empty(nmax)
        return StabilizerGroupEnergies(Ps, energy_level)

    def range(self, begin, end, allow_truncation=False):
        if not allow_truncation:
            if begin > self.Ps.begin:
                assert np.all(self.Ps.array4[:, 0:begin - self.Ps.begin] == 0)
            if end < self.Ps.end:
                assert np.all(self.Ps.array4[:, end - self.Ps.begin:] == 0)
        return StabilizerGroupEnergies(self.Ps.range(begin, end, allow_truncation=allow_truncation), self.energy_level, canonicalized=True)

    def flip_sign(self, index):
        self.Ps.ZX_phase[index] = (self.Ps.ZX_phase[index] + 2) % 4
        self.energy_level.flip_sign(index)

    def canonicalize_signs(self):
        for i in range(self.m):
            if self.Ps[i].phase == -1:
                self.flip_sign(i)

    def canonicalize(self):
        _, U = linear.Gauss_elimination_full(self.Ps.array, 2, allow_reorder=True)
        self.transform(U)
        self.canonicalize_signs()

    def copy(self):
        return StabilizerGroupEnergies(self.Ps.copy(), self.energy_level.copy(), canonicalized=True)

    def transform(self, U):
        Ps = self.Ps.copy()
        Ps.x = U.dot(self.Ps.x) % 2
        Ps.z = U.dot(self.Ps.z) % 2
        Ps.group_phase = 0
        signs = np.array([self.decompose(P)[1] for P in Ps], dtype=inttype)
        Ps = Ps * signs
        self.Ps = Ps

        self.energy_level.transform(U)

    def add(self, P: Pauli, index=-1) -> "StabilizerGroupEnergies":
        self = self.copy()
        self.Ps = Pauli.concatenate((self.Ps, P[None]))
        self.energy_level.add(index=index)
        self.initialize()
        self.canonicalize()
        return self

    def project_to_next(self) -> "StabilizerGroupEnergies":
        '''
        Before projection: stabilizer on the last k sites
        After projection: stabilizer on the last k-1 sites
        In canonicalized form, the array is automatically splitted into the subgroup and coset
        '''
        self = self.copy()
        while np.any(self.Ps.array4[:, 0] > 0):
            '''
            Simply remove the first term since it is already canonicalized, e.g.
            X Z Z
            I X X
            I I I
            '''
            self.Ps = self.Ps[1:]
            self.m -= 1
            self.energy_level = self.energy_level.project_to_next()
        self = self.range(self.Ps.begin + 1, self.Ps.end, allow_truncation=True)
        return self

    def project_to(self, k1) -> "StabilizerGroupEnergies":
        while self.begin < self.end - k1:
            self = self.project_to_next()
        return self

    def simplify_coset_representative(self, P: Pauli) -> Pauli:
        '''
        Assuming [P, S] = 0, try to find Q \in S such that P * Q is simplified (start with a large index)
        e.g. S = <XII, IYI>, P = XYZ, then Q = IIZ
        '''
        Ps, P = Pauli.extend(self.Ps, P)
        array = linear.simplify_coset_representative(Ps.array, P.array, 2)
        return Pauli.from_array(array, begin=P.begin)

    def add_energy(self, P: Pauli, w):
        x, sign = self.decompose(P)
        self.energy_level.add_energy(x, sign, w)

    @classmethod
    def merge_gs(cls, stab1, stab2) -> "StabilizerGroupEnergies":
        '''
        Use the lower energy as the energy of the merged state
        '''
        assert np.all(stab1.Ps.phase == 1)
        assert np.all(stab2.Ps.phase == 1)
        assert np.all(stab1.Ps.array4 == stab2.Ps.array4)
        Ps = stab1.Ps
        energy_level = EnergyLevel.merge_gs(stab1.energy_level, stab2.energy_level)
        return cls(Ps, energy_level, canonicalized=True)

    def shift(self, add):
        return StabilizerGroupEnergies(self.Ps.shift(add), self.energy_level, canonicalized=True)


def is_in_group(Ps: Pauli, P: Pauli):
    assert Ps.begin == P.begin
    assert Ps.end == P.end
    assert len(Ps.batch_size) == 1
    assert len(P.batch_size) == 0
    is_over, is_under, x = linear.solve_modN(Ps.array.T, P.array, 2)
    return (not is_over)
