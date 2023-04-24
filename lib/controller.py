import numpy as np
from lib.lqr import classical_lqr

def vanilla(a, pmd_slice, r2, q):
    n, _ = a.shape
    n_pmd = len(pmd_slice)
    b = np.zeros((n, n_pmd))
    b[np.ix_(pmd_slice, range(n_pmd))] = np.eye(n_pmd)
    return np.eye(n), np.dot(b, classical_lqr(r2, q, a, b))