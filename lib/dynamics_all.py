import numpy as np
from lib.defaults import Main
from lib.gramians import Make
import json
#I gave up trial, loop_gain and signals.off_on_off and use many if

class simulate_instantaneous_feedback(Main):
    def __init__(self):
        super().__init__()
        self.sample_every = round(self.sampling_dt / self.dt)
        self.spon = 0.1
        self.pre = 0.6
        self.mov = 0.6
        self.duration = self.spon + self.pre + self.mov
        self.n_bins = round(self.duration / self.dt)
        self.off1 = round(self.spon / self.dt)
        self.on = round(self.pre / self.dt)
        self.off2 = round(self.mov / self.dt)
 
    def run(self, loop, xstars, spontaneous, grad = False):        
        if grad:           
            import torch
            W = torch.Tensor(self.w_rec)
            nl = torch.nn.ReLU()
        else:                          
            W = self.w_rec  
            nl = self.nl 
        
        yx, xy = loop
        feedback = xy @ yx
        
        h = spontaneous - W @ nl(spontaneous)  
        specific_input = xstars - h - ((W + feedback) @ nl(xstars))
        accu = []
        t = 0
        x = np.repeat(spontaneous, xstars.shape[1], axis=1)
        
        while t < (self.n_bins):
            time = self.dt * t         
            r = nl(x)
            
            if t % self.sample_every == 0:
                accu.append(r)
                
            if t < self.off1:
                input_ = (h
                          + 0
                          + 0
                          + W @ r
                          + 0)
                
            if self.off1 <= t < self.off1+self.on:    
                input_ = (h
                          + 0
                          + specific_input
                          + W @ r
                          + feedback @ r)
                
            if self.off1+self.on <= t :    
                input_ = (h
                          + self.mov_input(time - self.spon - self.pre)
                          + 0
                          + W @ r
                          + 0)
            
            x = (1.0 - (self.dt / self.tau)) * x + (self.dt / self.tau) * input_
            
            t += 1

        return accu
