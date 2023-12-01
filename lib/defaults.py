import numpy as np

# main + base in OCaml and omit number of (phantom) muscles in my practice
class Base:
    def __init__(self):
        self.dt = 2e-4
        self.sampling_dt = 1e-3
        self.tau = 150e-3
        self.baseline_rate = 1.
        self.spontaneous_std = 0.15
        self.r2_vanilla = 0.1
        self.r2_instant = 0.1
        self.r2_dynamic = 0.01
        self.taus = 10e-3, 10e-3
        self.mov_input = self.alpha_bump(50, 500, 5)
        
    def alpha_bump(self, tau_rise, tau_decay, amplitude):
        tmax = np.log(tau_decay / tau_rise) * tau_decay * tau_rise / (tau_decay - tau_rise)
        def f(t):
            return np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
        amax = f(tmax)
        return lambda t: amplitude / amax * f(t) 
    
class Main(Base):
    def __init__(self):
        super().__init__()
        self.w_rec = np.loadtxt('isn.txt')
        self.n = self.w_rec.shape[0]
        self.a = self.w_rec - np.eye(self.n)
        self.n_e = round(0.8 * self.n)
        self.n_i = self.n - self.n_e
        self.m1_slice = list(range(self.n_e))
        self.pmd_slice = list(range(self.n))
        self.gamma = np.eye(self.n)[self.m1_slice]
    def nl(self,x):
        return np.maximum(0,x) 