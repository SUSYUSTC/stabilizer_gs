import numpy as np
import typing as tp
from pauli import Pauli
from stab import StabilizerGroup, StabilizerGroupEnergies, is_in_group
import tqdm

type_index = tp.Tuple[int, int]
tot_count = 0


class Hamiltonian:
    def __init__(self, n, k):
        '''
        Divide the Pauli terms by the last site (qlast)
        '''
        self.n = n
        self.k = k
        self.l = None
        self.original_paulis = []
        self.original_hashs = []
        self.paulis: tp.Dict[int, tp.List[tp.Tuple[Pauli, float]]] = {qlast: [] for qlast in range(n)}

    def add_Pauli(self, P: Pauli, w):
        assert P.end - P.begin <= self.k
        self.paulis[P.end-1].append((P, w))
        self.original_paulis.append((P, w))
        self.original_hashs.append(hash(P))

    def get_index(self, m, i):
        P, _ = self.paulis[m][i]
        return self.original_hashs.index(hash(P))
        # return i * self.n + self.m

    @classmethod
    def from_file(cls, path):
        import re
        f = open(path)
        n, k = map(int, f.readline()[0:-1].split(' '))
        H = Hamiltonian(n, k)
        while True:
            line = f.readline().strip()
            if not line:
                break
            args = re.split(r'\s+', line)
            w = float(args[0])
            string = args[1]
            qubits = list(map(int, args[2:]))
            H.add_Pauli(Pauli.from_string(string, qubits), w)
        return H

    @classmethod
    def from_file_periodic(cls, path, cmax):
        import re
        f = open(path)
        l, k = map(int, f.readline()[0:-1].split(' '))
        n = l * (1+cmax) + k * 2
        H = Hamiltonian(n, k)
        H.l = l
        while True:
            line = f.readline().strip()
            if not line:
                break
            args = re.split(r'\s+', line)
            w = float(args[0])
            string = args[1]
            qubits = np.array(list(map(int, args[2:])))
            qmax = np.max(qubits)
            qmin = np.min(qubits)
            for i in range(-n, n+1):
                if (qmax + i * l < n) and (qmin + i * l >= 0):
                    H.add_Pauli(Pauli.from_string(string, (qubits + i * l).tolist()), w)
        return H


def hash_dict(d):
    return hash(tuple(sorted(d.items())))


def update_dict(d, items, merge_func):
    for key, value in items:
        if key in d:
            d[key] = merge_func(d[key], value)
        else:
            d[key] = value
    return d


class FullStateEnergies:
    def __init__(self, paulis: tp.List[tp.Tuple[Pauli, float]]):
        n = np.max([p.end for p, w in paulis])
        self.stab = StabilizerGroupEnergies.empty(0, n)
        self.paulis = paulis
        self.excluded_paulis = []

    def copy(self):
        result = FullStateEnergies.__new__(FullStateEnergies)
        result.stab = self.stab.copy()
        result.paulis = [(P.copy(), w) for P, w in self.paulis]
        result.excluded_paulis = [P.copy() for P in self.excluded_paulis]
        return result

    def include_pauli(self, P: Pauli):
        self = self.copy()
        self.stab = self.stab.add(P)
        new_paulis = []
        for Q, w in self.paulis:
            if not self.stab.commute(Q):
                continue
            if self.stab.include(Q):
                self.stab.add_energy(Q, w)
            else:
                tmp_stab = self.stab.add(Q)
                if not np.any([tmp_stab.include(R) for R in self.excluded_paulis]):
                    new_paulis.append((Q, w))
        self.paulis = new_paulis
        return self

    def exclude_pauli(self, P: Pauli):
        self = self.copy()
        self.excluded_paulis.append(P)
        self.paulis = [(Q, w) for Q, w in self.paulis if not self.stab.add(Q).include(P)]
        return self

    def evolve(self):
        state = self.copy()
        states = [state]
        while True:
            finished = np.all([len(state.paulis) == 0 for state in states])
            if finished:
                break
            new_states = []
            for state in states:
                if len(state.paulis) == 0:
                    new_states.append(state)
                    continue
                P, _ = state.paulis[0]
                new_states.append(state.include_pauli(P))
                new_states.append(state.exclude_pauli(P))
            states = new_states
            print('number of states', len(states))
        states
        index_state = np.argmin([np.min(state.stab.energy_level.Es) for state in states])
        state_gs = states[index_state]
        energy_level = state_gs.stab.energy_level
        index_sign = np.unravel_index(np.argmin(energy_level.Es), energy_level.Es.shape)
        phase = 1 - np.array(index_sign) * 2
        return state_gs.stab.Ps * phase, energy_level.Es[index_sign]


class State:
    '''
    define {Pij|ij} s.t. qlast of Pij is i
    Pm = {Pmj|j}
    Given site m, a state is defined by:
    1. Sproj: projection of S(<m) to [m-k+1, m-1]
    2. valid indices of Ps s.t. qfirst <= m-1, qlast >= m.
    3. Sright: projection of S(>=m) to [m-k+1, m]
    assert: if P not in S, then P not in <S, Sright>
    possible: P in Sproj and P in Sright
    which is well defined at m=0 (Sproj is empty, valid indices are all)
    m' = m+1
    update m:
    1. Divide Pm to P_include and P_exclude depending on Sright
    2. Sproj -> projection of <Sproj, P_include> to [m'-k+1, m'-1]
    3. Update valid indices to qfirst <= m'-1, qlast >= m'
    4. Create branches Sright' in S(>=m') in [m'-k+1, m'] such that
        1. Sright' are generated by truncation of valid Ps
        2. projection of <Sright', P_include> to [m-k+1, m] is Sright
    proof:
    1. if Sright' is valid, then Sright is valid
    '''
    def __init__(self, hamiltonian: Hamiltonian, Sright, n=None):
        self.n = hamiltonian.n
        self.k = hamiltonian.k
        self.hamiltonian = hamiltonian
        self.m = 0
        if n is None:
            n = hamiltonian.n
        self.stab = StabilizerGroupEnergies.empty(0, n)
        #self.valid_pauli_indices: tp.Dict[int, tp.List[int]] = {qlast: list(range(len(self.hamiltonian.paulis[qlast]))) for qlast in range(self.n)}
        self.invalid_pauli_indices: tp.Dict[int, tp.List[int]] = {qlast: [] for qlast in range(self.n)}
        self.Sright = Sright

    def get_invalid_indices(self, qlast):
        return self.invalid_pauli_indices[qlast]

    def get_valid_indices(self, qlast):
        n = len(self.hamiltonian.paulis[qlast])
        return list(set(range(n)).difference(set(self.invalid_pauli_indices[qlast])))

    def get_invalid_Ps(self, qrange):
        return [self.hamiltonian.paulis[qlast][i][0] for qlast in qrange for i in self.get_invalid_indices(qlast)]

    def get_valid_Ps(self, qrange):
        return [self.hamiltonian.paulis[qlast][i][0] for qlast in qrange for i in self.get_valid_indices(qlast)]

    @property
    def invalid_paulis_all(self):
        return self.get_invalid_Ps(range(self.n))

    @property
    def invalid_range(self):
        #return range(self.m, min(self.m + self.k - 1, self.n))
        return range(self.m + 1, min(self.m + self.k - 1, self.n))

    @property
    def Sright_range(self):
        return range(self.m + 1, min(self.m + self.k + 1, self.n))

    @property
    def invalid_pauli_indices_part(self):
        '''
        Unique representation of valid paulis for hash
        '''
        return {qlast: tuple(sorted(self.invalid_pauli_indices[qlast])) for qlast in self.invalid_range}

    def __hash__(self):
        return hash((self.m, self.stab, hash_dict(self.invalid_pauli_indices_part), self.Sright))

    def copy(self) -> "State":
        from copy import deepcopy
        result = State(self.hamiltonian, self.Sright)
        result.m = self.m
        result.stab = self.stab.copy()
        result.invalid_pauli_indices = deepcopy(self.invalid_pauli_indices)
        return result

    def to_string(self):
        string1 = f"m = {self.m}\n"
        string2 = f"stab = {self.stab}\n"
        invalid_Ps = self.get_invalid_Ps(self.invalid_range)
        invalid_paulis = Pauli.stack(invalid_Ps) if len(invalid_Ps) > 0 else 'empty'
        string3 = f"Sright = {self.Sright}\n"
        string4 = f"invalid_pauli_indices = {invalid_paulis}\n"
        string5 = f"energies = {self.stab.energy_level.Es.flatten()}\n"
        string6 = f"Ps indices \n{self.stab.energy_level.Ps_indices} \n"
        string7 = f"Ps signs \n{self.stab.energy_level.Ps_signs} \n"
        #return string1 + string2 + string3 + string4 + '\n'
        return string1 + string2 + string3 + string4 + string5 + string6 + string7 + '\n'

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def check_Sright_validity(self, Sright_next):
        assert Sright_next.begin == max(self.m - self.k + 2, 0)
        assert Sright_next.end == self.m + 2
        valid_Ps = self.get_valid_Ps(self.Sright_range)
        begin = max(self.m - self.k + 2, 0)
        end = self.m + 2
        if len(valid_Ps) > 0:
            valid_Ps = Pauli.stack(valid_Ps).range(begin, end, allow_truncation=True)
        else:
            valid_Ps = Pauli.identity(end - begin, (0, ), begin=begin)
        for P in Sright_next.Ps:
            if not is_in_group(valid_Ps, P):
                return False
        return True

    def add(self, P, index=-1):
        self.stab = self.stab.add(P, index=index)
        #qlast = P.end - 1
        for tmp_qlast in range(P.begin, min(P.end - 1 + self.k, self.n)):
            '''
            left_terms = []
            for tmp_index in self.valid_pauli_indices[tmp_qlast]:
                tmp_pauli, tmp_weight = self.hamiltonian.paulis[tmp_qlast][tmp_index]
                if self.stab.commute(tmp_pauli):
                    left_terms.append(tmp_index)
            self.valid_pauli_indices[tmp_qlast] = left_terms
            '''
            for tmp_index in self.get_valid_indices(tmp_qlast):
                tmp_pauli, tmp_weight = self.hamiltonian.paulis[tmp_qlast][tmp_index]
                if not self.stab.commute(tmp_pauli):
                    self.invalid_pauli_indices[tmp_qlast].append(tmp_index)

    def move_to_next(self):
        #old_self = self
        self = self.copy()
        stab_tmp = StabilizerGroup(self.stab.Ps)
        for Pright in self.Sright.Ps:
            assert stab_tmp.commute(Pright)
            if not stab_tmp.include(Pright):
                stab_tmp = stab_tmp.add(Pright)
        for i, (P, w) in enumerate(self.hamiltonian.paulis[self.m]):
            #index = i * self.n + self.m
            index = self.hamiltonian.get_index(self.m, i)
            if not stab_tmp.include(P):
                continue
            if not self.Sright.include(P):
                return None
            assert self.stab.commute(P)
            if not self.stab.include(P):
                self.add(P, index=index)
        for i, (P, w) in enumerate(self.hamiltonian.paulis[self.m]):
            if self.stab.include(P):
                self.stab.add_energy(P, w)
        self.stab = self.stab.range(max(self.m - self.k + 1, 0), self.m + 1)
        return self

    @classmethod
    def merge_gs(cls, state1, state2: "State"):
        state = state1.copy()
        state.stab = StabilizerGroupEnergies.merge_gs(state1.stab, state2.stab)
        return state

    def shift(self, add, clear_history=False):
        self = self.copy()
        self.m = self.m + add
        self.stab = self.stab.shift(add)
        self.Sright = self.Sright.shift(add)
        invalid_pauli_indices = {qlast: [] for qlast in range(self.n)}
        for qlast, values in self.invalid_pauli_indices.items():
            if (qlast + add < 0) or (qlast + add) >= self.n:
                continue
            invalid_pauli_indices[qlast + add] = values
        self.invalid_pauli_indices = invalid_pauli_indices
        if clear_history:
            self.clear_history()
        return self

    def clear_history(self):
        self.stab.energy_level.clear()


class StateMachine:
    def __init__(self, hamiltonian: Hamiltonian, Sright_all):
        self.n = hamiltonian.n
        self.k = hamiltonian.k
        self.m = 0
        self.hamiltonian = hamiltonian
        self.state_dict = {}
        self.Sright_all = Sright_all
        Srights, _ = Sright_all[0]
        for Sright in Srights.values():
            state = State(hamiltonian, Sright)
            self.state_dict[hash(state)] = state

    def evolve_single(self, raw_state: State):
        state = raw_state.move_to_next()
        if state is None:
            return []
        state.stab = state.stab.project_to(self.k - 1)
        assert state.stab.begin == max(state.m - self.k + 2, 0)
        assert state.stab.end == self.m + 1
        for m in range(self.n):
            for i in state.get_valid_indices(m):
                P, w = self.hamiltonian.paulis[m][i]
                assert state.stab.commute(P)

        new_states = []
        _, Sright_next_ids_map = self.Sright_all[state.m]
        Sright_next_all, _ = self.Sright_all[state.m + 1]
        for index in Sright_next_ids_map[hash(state.Sright)]:
            Sright_next = Sright_next_all[index]
            new_state = state.copy()
            if not new_state.check_Sright_validity(Sright_next):
                continue
            new_state.Sright = Sright_next
            new_state.m += 1
            new_states.append(new_state)
        return new_states

    def evolve(self):
        new_state_dict = {}
        for state in tqdm.tqdm(self.state_dict.values()):
            new_states = self.evolve_single(state)
            for new_state in new_states:
                hash_value = hash(new_state)
                if hash_value in new_state_dict:
                    new_state_dict[hash_value] = State.merge_gs(new_state_dict[hash_value], new_state)
                else:
                    new_state_dict[hash_value] = new_state
        self.state_dict = new_state_dict
        self.m += 1

    def generate_gs(self):
        # find state & signs with the lowest energy
        state_list = list(self.state_dict.values())
        index_state = np.argmin([np.min(state.stab.energy_level.Es) for state in state_list])
        self.state_gs = state_list[index_state]
        level_gs = self.state_gs.stab.energy_level
        self.index_sign = np.unravel_index(np.argmin(level_gs.Es), level_gs.Es.shape)
        self.energy_gs = level_gs.Es[self.index_sign]
        nPs = level_gs.nPs[self.index_sign]
        Ps_index = level_gs.Ps_indices[self.index_sign][0:nPs]
        sign_index = level_gs.Ps_signs[self.index_sign][0:nPs]
        Ps = []
        '''
        for this_qlast, this_index, sign in zip(Ps_index % self.n, Ps_index // self.n, sign_index):
            P = self.hamiltonian.paulis[this_qlast][this_index][0]
            Ps.append(P * sign)
        '''
        for index, sign in zip(Ps_index, sign_index):
            P, _ = self.hamiltonian.original_paulis[index]
            Ps.append(P * sign)
        self.stab_gs = StabilizerGroup(Pauli.stack(Ps))


class StateMachinePeriodic:
    def __init__(self, hamiltonian: Hamiltonian, Sright_all):
        self.n = hamiltonian.n
        self.k = hamiltonian.k
        self.l = self.n - self.k + 1
        self.m = 0
        self.hamiltonian = hamiltonian
        self.state_dict = {}
        self.Sright_all = Sright_all
        Srights, _ = Sright_all[0]
        for Sright in Srights.values():
            # in case of error increase the 3 here
            state = State(hamiltonian, Sright, n=hamiltonian.n + self.l * 3)
            self.state_dict[hash(state)] = state

    def evolve_single(self, raw_state: State):
        state = raw_state.move_to_next()
        if state is None:
            return []
        state.stab = state.stab.project_to(self.k - 1)
        assert state.stab.begin == max(state.m - self.k + 2, 0)
        assert state.stab.end == self.m + 1
        for m in range(self.n):
            for i in state.get_valid_indices(m):
                P, w = self.hamiltonian.paulis[m][i]
                assert state.stab.commute(P)

        new_states = []
        _, Sright_next_ids_map = self.Sright_all[state.m]
        Sright_next_all, _ = self.Sright_all[state.m + 1]
        for index in Sright_next_ids_map[hash(state.Sright)]:
            Sright_next = Sright_next_all[index]
            new_state = state.copy()
            if not new_state.check_Sright_validity(Sright_next):
                continue
            new_state.Sright = Sright_next
            new_state.m += 1
            new_states.append(new_state)
        return new_states

    def evolve(self):
        new_state_dict = {}
        for state in self.state_dict.values():
            new_states = self.evolve_single(state)
            for new_state in new_states:
                hash_value = hash(new_state)
                if hash_value in new_state_dict:
                    new_state_dict[hash_value] = State.merge_gs(new_state_dict[hash_value], new_state)
                else:
                    new_state_dict[hash_value] = new_state
        self.state_dict = new_state_dict
        self.m += 1

    def generate_gs(self):
        # find state & signs with the lowest energy
        state_list = list(self.state_dict.values())
        index_state = np.argmin([np.min(state.stab.energy_level.Es) for state in state_list])
        self.state_gs = state_list[index_state]
        level_gs = self.state_gs.stab.energy_level
        index_sign = np.unravel_index(np.argmin(level_gs.Es), level_gs.Es.shape)
        self.stab_gs, self.energy_gs = level_gs.get_state(index_sign, self.hamiltonian.original_paulis)


type_branch = tp.Tuple[StabilizerGroup, tp.List[Pauli]]
type_Sright = tp.Tuple[tp.List[StabilizerGroup], tp.Dict[int, tp.List[int]]]


def branch_include(branch: type_branch, P: Pauli) -> type_branch:
    stab, exclude_paulis = branch
    stab = stab.add(P)
    for ep in exclude_paulis:
        if stab.include(ep):
            return None
    return (stab, exclude_paulis)


def branch_exclude(branch: type_branch, P: Pauli) -> type_branch:
    stab, exclude_paulis = branch
    exclude_paulis = [item.copy() for item in exclude_paulis]
    exclude_paulis.append(P)
    return (stab, exclude_paulis)


def evolve_stab(stab: StabilizerGroup, paulis: tp.List[Pauli], project=True) -> tp.List[StabilizerGroup]:
    begin = stab.begin
    end = stab.end
    for p in paulis:
        assert p.end == end - 1
    branches = [(stab, [])]
    for P in paulis:
        new_branches = []
        for branch in branches:
            stab, excluded_paulis = branch
            if stab.commute(P) and (not stab.include(P)):
                branch1 = branch_include(branch, P)
                branch2 = branch_exclude(branch, P)
                if branch1 is not None:
                    new_branches.append(branch1)
                new_branches.append(branch2)
            else:
                new_branches.append(branch)
        branches = new_branches
    begin_next = max(begin - 1, 0)
    stabs = [stab.range(begin_next, end) for stab, _ in branches]
    if project:
        stabs = [stab.project(begin_next, end - 1) for stab in stabs]
    return stabs


def reverse_map(map):
    ys = sum(list(map.values()), [])
    reverse_map = {y: [] for y in ys}
    for x, y_list in map.items():
        for y in y_list:
            reverse_map[y].append(x)
    return reverse_map


def get_map_from_stabs(stabs_next, stabs_dict_prev: tp.List[tp.List[StabilizerGroup]]):
    hashs_dict_prev = {key: [hash(item) for item in values] for key, values in stabs_dict_prev.items()}
    stab_maps = reverse_map(hashs_dict_prev)
    stab_maps = {key: set(values) for key, values in stab_maps.items()}
    stabs_prev = {hash(stab): stab for stab in sum(list(stabs_dict_prev.values()), [])}
    return stabs_prev, stab_maps


def generate_Sright_all(H: Hamiltonian) -> tp.Dict[int, type_Sright]:
    stab0 = StabilizerGroup.empty(H.k, begin=H.n - H.k + 1)
    stabs = {hash(stab0): stab0}
    Sright_all = {H.n: (stabs, {hash(stab): [] for stab in stabs.values()})}
    m = H.n - 1
    for m in range(H.n)[::-1]:
        paulis = [P.copy() for P, w in H.paulis[m]].copy()
        stabs_dict = {key: evolve_stab(stab, paulis) for key, stab in stabs.items()}
        stabs, stabs_maps = get_map_from_stabs(stabs, stabs_dict)
        Sright_all[m] = (stabs, stabs_maps)
        print(m, len(stabs))
    return Sright_all


def generate_Sright_periodic(H, cmax) -> tp.Dict[int, type_Sright]:
    end = H.l * (1+cmax) + H.k
    stab0 = StabilizerGroup.empty(H.k, begin=end - H.k + 1)  # l+1 ~ l+k
    stabs = {hash(stab0): stab0}
    Sright_all = {}
    id = None
    while True:
        for m in range(end - H.l, end)[::-1]:
            paulis = [P.copy() for P, w in H.paulis[m].copy()]
            stabs_dict = {key: evolve_stab(stab, paulis) for key, stab in stabs.items()}
            stabs, stabs_maps = get_map_from_stabs(stabs, stabs_dict)
            print(m, len(stabs))
        new_id = tuple(np.sort(list(stabs.keys())))
        # after this the key is at zero
        stabs = {key: value.shift(H.l) for key, value in stabs.items()}
        if new_id == id:
            break
        else:
            id = new_id
    for m in range(end)[::-1]:
        paulis = [P.copy() for P, w in H.paulis[m].copy()]
        stabs_dict = {key: evolve_stab(stab, paulis) for key, stab in stabs.items()}
        stabs, stabs_maps = get_map_from_stabs(stabs, stabs_dict)
        Sright_all[m] = (stabs, stabs_maps)
        print(m, len(stabs))
    return Sright_all
