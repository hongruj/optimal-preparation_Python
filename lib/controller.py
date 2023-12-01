import numpy as np
from lib.lqr import classical_lqr, output_lqr
import torch
from torch import optim

def vanilla(a, pmd_slice, r2, q):
    n, _ = a.shape
    n_pmd = len(pmd_slice)
    b = np.zeros((n, n_pmd))
    b[np.ix_(pmd_slice, range(n_pmd))] = np.eye(n_pmd)
    return np.eye(n), np.dot(b, classical_lqr(r2, q, a, b))



def dynamic(a, pmd_slice, n_e, tau, taus, r2, q):
    n, _ = a.shape
    n_pmd = len(pmd_slice)
    n_i = n - n_e
    tau_z, tau_y = taus
    const_y = (tau / tau_y) * np.eye(n_e)
    const_z = (tau / tau_z) * np.eye(n_e)

    yx = 0.1
    x = np.random.binomial(1, yx, size=(n_e, n_e)) / (yx * n_e)
    yx = np.hstack((x, np.zeros((n_e, n_i))))

    a = np.block([
        [a, np.zeros((n, n_e)), np.zeros((n, n_e))],
        [(tau / tau_y) * yx, -const_y, np.zeros((n_e, n_e))],
        [np.zeros((n_e, n)), const_z, -const_z]
    ])

    b = np.block([
        [np.eye(n)],
        [np.zeros((n_e + n_e, n))]
    ])

    c = np.hstack((np.zeros((n_e, n)), np.zeros((n_e, n_e)), np.eye(n_e)))

    q = np.block([
        [q, np.zeros((n, n_e)), np.zeros((n, n_e))],
        [np.zeros((n_e, n)), np.zeros((n_e, n_e)), np.zeros((n_e, n_e))],
        [np.zeros((n_e, n)), np.zeros((n_e, n_e)), np.zeros((n_e, n_e))]
    ])

    xy = output_lqr(r2, q, a, b, c, (n_pmd, n_e))
    return yx, b @ xy


# I found loss smaller in pytorch than scipy.optimize.minimize (res.fun is too large sometimes)
def decompose(n, n_z, pmd_slice, yx_xy, reg):
    yx0, xy0 = yx_xy
    yx = torch.Tensor(yx0)
    xy = torch.Tensor(xy0)
    
    xy = xy[pmd_slice, :]
    n_pmd, n_e = xy.shape
    n_z_e = int(0.5 * n_z)
    n_prms = (n_pmd * n_z) + (n_z * n_e)
    lambda_ = 1. / torch.linalg.norm(xy) ** 2
    reg = reg / n_prms

    def for_zy(prms):
        return prms[:n_z * n_e].reshape(n_z, n_e)

    def for_xz(prms):
        return prms[n_z * n_e:].reshape(n_pmd, n_z) 

    def f_new(prms):
        zy_prms = for_zy(prms)
        xz_prms = for_xz(prms)
        zy = zy_prms ** 2
        xz = torch.hstack((xz_prms[:, :n_z_e] ** 2, -xz_prms[:, n_z_e:] ** 2))
        return zy, xz

    def cost_fun(prms):
        zy, xz = f_new(prms)
        cost = lambda_ * torch.linalg.norm(xz @ zy - xy) ** 2 + reg * (torch.linalg.norm(zy) ** 2 + torch.linalg.norm(xz) ** 2)
        return cost

    def train(n_iter): 
    # torch  
        updater = optim.Adam([prms0])  
        for i in range(n_iter):                        
            l = cost_fun(prms0)
            updater.zero_grad()
            l.backward()
            updater.step()
            if (i+1) % (n_iter // 3) == 0 or i==0:
                print(f'iteration {i+1}/{n_iter} | train loss: {l.item():.6f}')
            if l.item() < 1e-6:
                print(f'train loss: {l.item():.6f}')
                break
        return prms0


    # Initial parameter vector
    prms0 = torch.normal(0, 0.1, size=(n_prms,)).requires_grad_(True)

    # Optimization
    prms=train(3000)

    # Get results
    zy, xz = f_new(prms)

    # Equalize the Frobenius norms
    zy_ = torch.linalg.norm(zy)
    xz_ = torch.linalg.norm(xz)
    yx_ = torch.linalg.norm(yx)
    zy = torch.sqrt(xz_ * yx_) / zy_* zy
    xz = torch.sqrt(zy_ * yx_) / xz_* xz
    yx = torch.sqrt(zy_ * xz_) / yx_ * yx

    # Combine results
    full_xz = torch.zeros((n, xz.shape[1]))
    full_xz[pmd_slice, :] = xz

    return full_xz, zy, yx