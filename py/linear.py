import numpy as np


class UnderDetermined(Exception):
    pass


class OverDetermined(Exception):
    pass


def get_inv_list(N, dtype=None):
    x = np.arange(N, dtype=dtype)
    return np.argmax((x[:, None] * x) % N == 1, axis=1)


def get_order(A):
    n, m = A.shape[-2:]
    zero = np.sum(A, axis=-1) == 0
    order = np.argsort(np.argmax(A > 0, axis=-1) + zero * m, axis=-1, kind='stable')
    return order


def Gauss_elimination(A, b, N, allow_reorder=False, verbose=False):
    # convert A by row transformation to
    # I C
    # O O
    if np.prod(A.shape) == 0:
        return A.copy(), b.copy()
    A = A.copy()
    b = b.copy()
    inv_list = get_inv_list(N, dtype=A.dtype)

    n, m = A.shape[-2:]
    shape = A.shape[0:-2]
    indices = tuple(np.indices(shape))

    def div(a, b):
        return (a * inv_list[b]) % N

    def update(A, b, k, pivot, slic):
        f = div(A[indices + (slic, pivot)], A[indices + (slice(k, k+1), pivot)])
        A[..., slic, :] = (A[..., slic, :] - A[..., k:k+1, :] * f[..., :, None]) % N
        b[..., slic, :] = (b[..., slic, :] - b[..., k:k+1, :] * f[..., :, None]) % N

    if verbose:
        print(np.concatenate((A, b), axis=-1))
        print()

    pivots = []
    pivot = 0
    for k in range(n):
        pivot = np.argmax(A[..., k, :] > 0, axis=-1)  # ..., 0-m
        slic = slice(k+1, n)
        update(A, b, k, pivot, slic)
        if verbose:
            print(np.concatenate((A, b), axis=-1))
            print()
        pivots.append(pivot)

    for k in range(n - 1, -1, -1):
        pivot = pivots[k]  # ..., 0-m
        slic = slice(0, k)
        update(A, b, k, pivot, slic)
        inv = inv_list[A[indices + (k, pivot)]]

        do_update = (inv[..., None] != 0)
        A_k_update = (A[..., k, :] * inv[..., None]) % N
        b_k_update = (b[..., k, :] * inv[..., None]) % N
        A[..., k, :] = A_k_update * do_update + A[..., k, :] * np.logical_not(do_update)
        b[..., k, :] = b_k_update * do_update + b[..., k, :] * np.logical_not(do_update)
        if verbose:
            print(np.concatenate((A, b), axis=-1))
            print()
    if allow_reorder:
        indices_new = tuple([item[..., None] for item in indices])
        order = get_order(A)
        A = A[indices_new + (order, )]
        b = b[indices_new + (order, )]
    return A, b


def Gauss_elimination_Aonly(A, N, allow_reorder=False, verbose=False):
    n, m = A.shape[-2:]
    b = np.zeros(A.shape[0:-2] + (n, 0), dtype=A.dtype)
    return Gauss_elimination(A, b, N, allow_reorder=allow_reorder, verbose=verbose)[0]


def Gauss_elimination_full(A, N, allow_reorder=False, verbose=False):
    n, m = A.shape[-2:]
    b = np.zeros(A.shape[0:-2] + (n, n), dtype=A.dtype)
    arange = np.arange(n)
    b[..., arange, arange] = 1
    return Gauss_elimination(A, b, N, allow_reorder=allow_reorder, verbose=verbose)


def get_col_order(arg, N):
    # step1: array = [4, 5, 6, 7]
    # step2: array = [4, 0, 6, 1]
    # step3: order = [1, 3, 0, 2]
    shape = arg.shape[0:-1]
    n = arg.shape[-1]
    indices = tuple(np.indices(shape))
    indices_new = tuple([item[..., None] for item in indices])

    array = np.broadcast_to(np.arange(N, N * 2), shape + (N, )).copy()
    array[indices_new + (arg, )] = np.broadcast_to(np.arange(n), shape + (n, ))
    return np.argsort(array, axis=-1)


def get_kernel_reordered(A2, N):
    # assume A2 is obtained by Gaussian_elimination(xxx, allow_reorder=True)
    A2 = A2[np.sum(A2, axis=1) > 0]
    n, m = A2.shape[-2:]
    if m == 0:
        return np.zeros((0, 0), dtype=int)
    assert n <= m
    shape = A2.shape[0:-2]
    indices = tuple(np.indices(shape))
    indices_new = tuple([item[..., None] for item in indices])

    assert np.all(np.any(A2, axis=-1))

    col_order = get_col_order(np.argmax(A2 > 0, axis=-1), m)
    # A2_sort: reorder columns of A so that A2_sort = (I, C), I: nxn, C: nx(m-n)
    # x = (a, b)^T, a: nx(m-n), b: (m-n)x(m-n), x: mx(m-n)
    # b = I, a = -Cb = -C
    A2_sort = A2[indices_new + (slice(None), col_order)]
    b_sort = np.broadcast_to(np.eye(m - n, dtype=A2.dtype), shape + (m - n, m - n))
    a_sort = (- A2_sort[..., :, n:]) % N
    x_sort = np.concatenate((a_sort, b_sort), axis=-2)  # (..., m, m - n)
    x = np.zeros_like(x_sort)
    x[indices_new + (col_order, slice(None))] = x_sort
    return x


def get_kernel(A, N):
    # linear space: {x|Ax=0 (mod N)}
    A2 = Gauss_elimination_Aonly(A, N, allow_reorder=True)
    return get_kernel_reordered(A2, N)


def get_kernel_reordered_bool(A2):
    return get_kernel_reordered(A2.astype(int), 2).astype(bool)


def get_kernel_bool(A):
    return get_kernel(A.astype(int), 2).astype(bool)


def solve_modN(A, b, N, verbose=False):
    if b.ndim == 1:
        is_over, is_under, x = solve_modN(A, b[:, None], N)
        return is_over[0], is_under, x[:, 0]
    n, m = A.shape
    assert b.shape[0] == n
    k = b.shape[1]

    if m == 0:
        is_over = np.any(b > 0, axis=0)
        is_under = False
        x = np.zeros((m, k), dtype=int)
        return is_over, is_under, x

    A2, b2 = Gauss_elimination(A, b, N, verbose=verbose)
    conditions = np.sum(A2, axis=1)
    is_over = np.any(b2[conditions == 0] > 0, axis=0)
    is_under = np.any(conditions > 1)
    A2 = A2[conditions > 0]
    b2 = b2[conditions > 0]
    pivots = np.argmax(A2, axis=1)
    x = np.zeros((m, k), dtype=int)
    x[pivots] = b2.copy()
    return is_over, is_under, x


def simplify_coset_representative(A2, x, N):
    if np.prod(A2.shape) == 0:
        return x
    argmax = np.argmax(A2 > 0, axis=-1)
    assert np.all(A2[np.arange(len(A2)), argmax] == 1)
    return (x - x[argmax].dot(A2)) % N


if __name__ == '__main__':
    N = 3
    n = 4
    A = np.random.randint(0, N, size=(5, n, n))
    b = np.zeros_like(A)
    b[:, np.arange(n), np.arange(n)] = 1
    A2, b2 = Gauss_elimination(A, b, N)
    assert np.all(np.einsum("xij,xjk->xik", b2, A) % N == A2)
