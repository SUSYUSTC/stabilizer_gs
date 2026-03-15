import numpy as np
import quimb as qu
import quimb.tensor as qtn
import functools


X = qu.spin_operator('X') * 2
Y = qu.spin_operator('Y') * 2
Z = qu.spin_operator('Z') * 2

conv = {"X":X,"Y":Y,"Z":Z}
def get_term(L, ps, sites):
    mpo = qtn.MatrixProductOperator.from_fill_fn(lambda shape: np.eye(2).reshape(shape), L, 1)
    for p, site in zip(ps, sites):
        tensor = mpo.tensors[site]
        data = p.reshape(tensor.shape)
        tensor.modify(data=data)
    return mpo


def get_norm(mpo1, mpo2):
    return np.linalg.norm((mpo1 | mpo2.conj()).contract() / (mpo1.norm() * mpo2.norm()))


def combine_mpos(mpos):
    mpo_full = functools.reduce(lambda x, y: x + y, mpos)
    mpo_full_compressed = mpo_full.compress_all_1d()
    norm = get_norm(mpo_full, mpo_full_compressed)
    return mpo_full_compressed, norm

def convert_to_mpos(paulis,coeffs):
    L = len(paulis[0])
    mpos_single = []
    mpos_double = []
    mpos_triple = []
    for (p,c) in zip(paulis,coeffs):
        data = [(i,x) for i,x in enumerate(p) if x != '_']
        if(len(data) == 3):
            term = get_term(L, [conv[d[1]] for d in data], [d[0] for d in data])
            mpos_triple.append(c * term)
        elif(len(data) == 2):
            term = get_term(L, [conv[d[1]] for d in data], [d[0] for d in data])
            mpos_double.append(c * term)
        else:
            term = get_term(L, [conv[d[1]] for d in data], [d[0] for d in data])
            mpos_single.append(c * term)

    if(len(mpos_triple)==0):
        return (combine_mpos(mpos_single)[0] + combine_mpos(mpos_double)[0]) 
    else:
        return (combine_mpos(mpos_single)[0] + combine_mpos(mpos_triple)[0])


def run_dmrg(mpos,D):
    dmrg = qtn.DMRG2(mpos, bond_dims=D, cutoffs=1e-10)
    dmrg.solve(tol=1e-6, verbosity=0, max_sweeps=50)
    return dmrg.energy