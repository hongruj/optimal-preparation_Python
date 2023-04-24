import numpy as np



def unit_trace(q):
    n = q.shape[0]
    z = n / np.trace(q)
    return z * q