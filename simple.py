import numpy as np
from lib.defaults import Main
import json
from lib.controller import vanilla
from lib.gramians import Make
from lib.dynamics_all import D

class Sim(Main): 
    def __init__(self):
        super().__init__()
        
        # Read the setup files
    def read_setup(self, inst=True):
        spontaneous = np.loadtxt('spontaneous.txt')
        if inst:
            prms = json.load(open("setup_prms.json"))
        else:
            prms = json.load(open("setup_prms_dynamics.json"))
        return spontaneous, np.array(prms['xstars']), np.array(prms['c'])    

    def q_mov(self, c):
        G = Make(a = self.a, gamma = c @ self.gamma)
        q = G.OSub.m_norm
# #        Preparing in the nullspace
#         q = 0.2*q + 0.8*200*(c.T@c)/np.linalg.norm(c.T@c)
        # Symmetrize for numerical stability
        return 0.5 *(q + q.T)   
    
    def vanilla(self, q, xstars, spontaneous):
        lqr_feedback = vanilla(self.a, self.pmd_slice, self.r2_vanilla, q)
        sif = D()
        r = sif.simulate_instantaneous_feedback(lqr_feedback, xstars, spontaneous)          
        return r, sif.duration
    
    def dynamic(self, xstars, spontaneous):    
        prms = json.load(open("loop_prms.json"))
        xz, zy, yx = np.array(prms['xz']), np.array(prms['zy']), np.array(prms['yx'])     
        loop = (xz, zy, yx)
        sif = D()
        x, y, z = sif.simulate_dynamic_feedback(loop, xstars, spontaneous)              
        return x, y, z, sif.duration 
     