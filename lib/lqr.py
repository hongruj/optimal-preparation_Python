import numpy as np
from scipy.linalg import solve_continuous_are as care
from scipy.linalg import solve_lyapunov, solve
from scipy.optimize import minimize

# VANILLA FULL-STATE LQR  
def classical_lqr(r2, q, a, b):
    m = b.shape[1]
    r = r2 * np.eye(m)
#     q_ = 0.2*q +0.8*m*(C.T@C)/np.trace(C.T@C)  # supplement  
    x = care(a, b, q, r)
    return - np.matmul(b.T, x) / r2



# OUTPUT LQR 
def output_lqr(r2, q, a, b, c, z_shape):
    n, _ = a.shape
    iden = np.eye(n)
    clo, raw = z_shape
    npara = clo * raw
    bt = b.T
    ct = c.T

    def k_of(z):
        return z @ c

    def s_of(acl):
        return solve_lyapunov(acl, -iden)

    def p_of(acl, k):
        return solve_lyapunov(acl.T, -(q + (r2 * k.T @ bt @ b @ k)))

    def grad_z(k, p, s):
        return 2 * bt @ (p + (r2 * b @ k)) @ s @ ct   

    
    def f_df(z):
        z = np.reshape(z, (clo, raw))
        k = k_of(z)
        acl = a + (b @ k) 
        p = p_of(acl, k)  
        return np.trace(p)

    def grad_func(z):
        z = np.reshape(z, (clo, raw))
        k = k_of(z)
        acl = a + (b @ k) 
        s = s_of(acl)
        p = p_of(acl, k)
        g_ = grad_z(k, p, s)    
        return g_.reshape(npara)

    z0 = classical_lqr(r2, q, a, b) @ solve(c @ c.T, c).T
    
    res = minimize(f_df, z0.reshape(npara), jac=grad_func, method='L-BFGS-B')
    return (res.x).reshape((clo, raw))