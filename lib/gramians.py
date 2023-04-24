import numpy as np
from scipy.linalg import solve_lyapunov


# No call CSub in the OCaml codes, so I omit it
class G:
    def __init__(self, m, m_norm, evals, evecs, get, top, bottom):
        self.m = m
        self.m_norm = m_norm
        self.evals = evals
        self.evecs = evecs
        self.get = get
        self.top = top
        self.bottom = bottom

        
class Make:
    def __init__(self, a, gamma=None):  
        self.a = a
        self.n = a.shape[0]

        self.O = G(*self.get("T", "I"))
        self.C = G(*self.get("N", "I"))

        rhs = "I" if gamma is None else np.dot(gamma.T, gamma)
        self.OSub = G(*self.get("T", rhs))

        
    def get(self, trans, rhs):
        rhs = -np.eye(self.n) if rhs is "I" else -rhs
        z = solve_lyapunov(self.a.T if trans == "T" else self.a, rhs)
        z_norm = self.n / np.trace(z) * z
        evecs, evals, _ = np.linalg.svd(z)
        get = lambda i: evecs[:, i]
        top = lambda k: evecs[:, :k]
        bottom = lambda k: evecs[:, self.n - k:]
        return z, z_norm, evals.T, evecs, get, top, bottom
