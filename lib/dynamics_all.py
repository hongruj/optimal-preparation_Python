import numpy as np
from lib.defaults import Main
from lib.gramians import Make
import json
#I gave up trial, loop_gain and signals.off_on_off and use many if

class D(Main):
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
 
    def simulate_instantaneous_feedback(self, loop, xstars, spontaneous):        
                          
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



    
    def simulate_dynamic_feedback(self, loop, xstars, spontaneous):            
        
        W = self.w_rec  
        nl = self.nl         
        xz, zy, yx = loop
        
        feedback = xz @ zy @ yx
        h = spontaneous - W @ nl(spontaneous)
        specific_input = xstars - h - ((W + feedback) @ nl(xstars))

        tau_z, tau_y = self.taus
        xaccu = []
        yaccu = []
        zaccu = []
        
        y = np.zeros((yx.shape[0], xstars.shape[1]))
        z = np.zeros((zy.shape[0], xstars.shape[1]))   
        x = np.repeat(spontaneous, xstars.shape[1], axis=1)
        
        t = 0
        while t < (self.n_bins):
            time = self.dt * t         
            rx = nl(x)
            ry = nl(y)
            rz = nl(z)
            
            if t % self.sample_every == 0:
                xaccu.append(rx)
                yaccu.append(ry)
                zaccu.append(rz)
                
                
            if t < self.off1:
                input_ = (h
                      + 0
                      + 0
                      + W @ rx
                      + xz @ rz)

            if self.off1 <= t < self.off1+self.on:    
                input_ = (h
                      + 0
                      + specific_input
                      + W @ rx
                      + xz @ rz)

            if self.off1+self.on <= t :    
                input_ = (h
                      + self.mov_input(time - self.spon - self.pre)
                      + 0
                      + W @ rx
                      + xz @ rz)
            
            x = (1.0 - (self.dt / self.tau)) * x + (self.dt / self.tau) * input_
            z = (1 - self.dt / tau_z) * z + (self.dt / tau_z) * zy @ ry
            if self.off1 <= t < self.off1+self.on:    
                y = (1 - self.dt / tau_y) * y + (self.dt / tau_y) * (yx @ rx)
            else:
                y = (1 - self.dt / tau_y) * y 
            
            t += 1

        return xaccu, yaccu, zaccu    
