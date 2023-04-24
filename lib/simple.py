import numpy as np
from lib.defaults import Main
import json
from lib.controller import vanilla
from lib.gramians import Make
from lib.dynamics_all import simulate_instantaneous_feedback

class Sim(Main): 
    def __init__(self):
        super().__init__()
        
        # Read the setup files
    def read_setup(self):
        spontaneous = np.loadtxt('spontaneous.txt')
        prms = json.load(open("setup_prms.json"))
        return spontaneous, np.array(prms['xstars']), np.array(prms['c'])    

    def q_mov(self, c):
        G = Make(a = self.a, gamma = c @ self.gamma)
        q = G.OSub.m_norm
        # Symmetrize for numerical stability
        return 0.5 *(q + q.T)   
    
    def vanilla(self, q, xstars, spontaneous):
        lqr_feedback = vanilla(self.a, self.pmd_slice, self.r2_vanilla, q)
        sif = simulate_instantaneous_feedback()
        r = sif.run(lqr_feedback, xstars, spontaneous)          
        return r, sif.duration