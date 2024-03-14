import numpy as np
from utils import zerosint2, randomint2
from functools import reduce


class Pauli:
    str_mapping = {
        'I': 0,
        'X': 1,
        'Y': 3,
        'Z': 2,
    }

    def __init__(self, z, x, begin=0, phase=None):
        assert z.shape == x.shape
        self.begin = begin
        self.length = z.shape[-1]
        self.end = self.begin + self.length
        self.batch_size = z.shape[0:-1]
        self.z = z % 2
        self.x = x % 2
        if phase is None:
            self.group_phase = zerosint2(self.batch_size)
        else:
            assert np.all(np.abs(phase) == 1)
            self.group_phase = 1 - phase

    @classmethod
    def dot_elementwise(cls, x, y):
        return np.sum(x * y, axis=-1)

    @property
    def nY(self):
        return self.dot_elementwise(self.z, self.x)

    @property
    def group_phase(self):
        return (self.ZX_phase + self.nY) % 4

    @group_phase.setter
    def group_phase(self, value):
        self.ZX_phase = (value - self.nY) % 4

    @property
    def phase(self):
        # (1j) ** self.group_phase
        assert np.all(np.abs(self.group_phase - 1) == 1)
        return 1 - self.group_phase

    @property
    def array(self):
        return np.stack([self.z, self.x], axis=-1).reshape(self.batch_size + (self.length * 2, ))

    @property
    def array4(self):
        return self.z * 2 + self.x

    def copy(self):
        return Pauli(self.z, self.x, begin=self.begin, phase=self.phase)

    def shift(self, add):
        return Pauli(self.z, self.x, begin=self.begin + add, phase=self.phase)

    @classmethod
    def from_array(cls, array, begin=0, phase=None) -> "Pauli":
        return Pauli(array[..., 0::2], array[..., 1::2], begin=begin, phase=phase)

    @classmethod
    def identity(cls, length, size=(), begin=0):
        shape = size + (length, )
        z = zerosint2(shape)
        x = zerosint2(shape)
        return Pauli(z, x, begin=begin)

    @classmethod
    def empty(cls, length, begin=0):
        return Pauli.identity(length, (0, ), begin)

    @classmethod
    def random(cls, length, size=(), begin=0):
        shape = size + (length, )
        z = randomint2(shape)
        x = randomint2(shape)
        phase = 1 - randomint2(size) * 2
        return Pauli(z, x, begin=begin, phase=phase)

    def __getitem__(self, key):
        return Pauli(self.z[key], self.x[key], begin=self.begin, phase=self.phase[key])

    def __mul__(self, c):
        result = self.copy()
        assert np.all(np.abs(c) == 1)
        add_phase = 1 - c
        result.ZX_phase = (result.ZX_phase + add_phase) % 4
        return result

    @classmethod
    def from_string(cls, string, sites):
        begin = np.min(sites)
        end = np.max(sites) + 1
        length = end - begin
        Z = zerosint2(length)
        X = zerosint2(length)
        for letter, site in zip(string, sites):
            value = cls.str_mapping[letter]
            z = value // 2
            x = value % 2
            Z[site - begin] = z
            X[site - begin] = x
        return cls(Z, X, begin=begin)

    def to_string(self):
        array4 = np.array(['I', 'X', 'Z', 'Y'])[self.array4]
        phase = np.array(['+', '-'])[(1 - self.phase) // 2]
        title = f'qubit {self.begin}-{self.end-1} size {self.batch_size} '
        if np.prod(self.batch_size) == 0:
            return title
        str_array = reduce(np.char.add, [phase] + np.moveaxis(array4, -1, 0).tolist()).flatten()
        return title + ' '.join(str_array)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def hash(self, add_position=True):
        if add_position:
            return hash((self.z.tobytes(), self.x.tobytes(), self.begin, self.phase.tobytes()))
        else:
            return hash((self.z.tobytes(), self.x.tobytes(), self.phase.tobytes()))

    def __hash__(self):
        return self.hash()

    def range(self, begin, end, allow_truncation=False):
        '''
        return a new pauli with range [begin, end) without modifying the operator value
        '''
        if not allow_truncation:
            assert begin <= self.begin
            assert end >= self.end
        length_begin = max(self.begin - begin, 0)
        length_end = max(end - self.end, 0)
        start = max(begin - self.begin, 0)
        stop = min(self.end - self.begin, end - self.begin)
        slic = slice(start, stop)
        zeros_begin = zerosint2(self.batch_size + (length_begin, ))
        zeros_end = zerosint2(self.batch_size + (length_end, ))
        z = np.concatenate([zeros_begin, self.z[..., slic], zeros_end], axis=-1)
        x = np.concatenate([zeros_begin, self.x[..., slic], zeros_end], axis=-1)
        return Pauli(z, x, begin=begin, phase=self.phase)

    def simplify(self) -> "Pauli":
        array4 = self.array4
        begin = self.begin
        end = self.end
        while np.all(array4[..., 0] == 0):
            begin += 1
            array4 = array4[..., 1:]
        while np.all(array4[..., -1] == 0):
            end -= 1
            array4 = array4[..., 0:-1]
        return self.range(begin, end, allow_truncation=True)

    @classmethod
    def extend(cls, *paulis):
        begin = np.min([p.begin for p in paulis])
        end = np.max([p.end for p in paulis])
        return [p.range(begin, end) for p in paulis]

    @classmethod
    def shape_operation(cls, operation, pauli_list, axis=0):
        pauli_list = cls.extend(*pauli_list)
        begin = pauli_list[0].begin
        z = operation([p.z for p in pauli_list], axis=axis)
        x = operation([p.x for p in pauli_list], axis=axis)
        phase = operation([p.phase for p in pauli_list], axis=axis)
        return Pauli(z, x, begin=begin, phase=phase)

    @classmethod
    def concatenate(cls, pauli_list, axis=0):
        return cls.shape_operation(np.concatenate, pauli_list, axis=axis)

    @classmethod
    def stack(cls, pauli_list, axis=0):
        return cls.shape_operation(np.stack, pauli_list, axis=axis)

    def commute(first, second: "Pauli"):
        first, second = Pauli.extend(first, second)
        count = Pauli.dot_elementwise(first.z, second.x) + Pauli.dot_elementwise(second.z, first.x)
        return (count % 2) == 0

    def dot(first, second: "Pauli"):
        assert np.all(first.commute(second))
        first, second = Pauli.extend(first, second)
        z = (first.z + second.z) % 2
        x = (first.x + second.x) % 2
        ZX_phase = (first.ZX_phase + second.ZX_phase + Pauli.dot_elementwise(first.x, second.z) * 2) % 4
        result = Pauli(z, x, begin=first.begin)
        result.ZX_phase = ZX_phase
        return result

    def reduce_dot(self):
        return reduce(Pauli.dot, self)


def test():
    Ps = Pauli.random(20, (10, ), begin=0)
    Qs = Pauli.random(20, (10, ), begin=5)
    valid = Ps.commute(Qs)
    Ps = Ps[valid]
    Qs = Qs[valid]
    result = Ps.dot(Qs).dot(Qs).dot(Ps)
    assert np.all(result.array4 == 0)
    assert np.all(result.phase == 1)


if __name__ == "__main__":
    test()
