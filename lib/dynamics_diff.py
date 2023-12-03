#  dynamics of the movement phase
import numpy as np
from lib.defaults import Main


class Dynamics(Main):
    def __init__(self):
        super().__init__()
        self.sample_every = round(self.sampling_dt / self.dt)
        

    def run(self, t_max, xstars, spontaneous, grad = False, layers=None):
        if grad:           
            import torch
            W = torch.Tensor(self.w_rec)
            nl = torch.nn.ReLU()
        else:                          
            W = self.w_rec  
            nl = self.nl 
            
        h = spontaneous - W @ nl(spontaneous)  
        accu = []
        t = 0
        x = xstars
        if layers:
            (tau_z, xz, z), (tau_y, zy, y) = layers
        
        while t < (t_max/self.dt):
            time = self.dt * t         
            r = nl(x)
            
            if t % self.sample_every == 0:
                accu.append(r)   
                
            input_sum = self.mov_input(time) + h + W @ r - x
            
            if layers:
                input_sum = input_sum + xz @ nl(z)
                y = (1 - self.dt / tau_y) * y
                z = (1 - self.dt / tau_z) * z + self.dt / tau_z * torch.mm(zy, nl(y)) 
#                 print(y)
            x = x + (self.dt/self.tau) * input_sum

            t += 1

        return accu
