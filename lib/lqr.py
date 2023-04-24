import numpy as np
from scipy.linalg import solve_continuous_are as care


# VANILLA FULL-STATE LQR  
def classical_lqr(r2, q, a, b):
    m = b.shape[1]
    r = r2 * np.eye(m)
#     q_ = 0.2*q +0.8*m*(C.T@C)/np.trace(C.T@C)  # supplement  
    x = care(a, b, q, r)
    return - np.matmul(b.T, x) / r2
