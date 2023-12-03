# optimal-preparation_Python

Python version codes of ["Optimal anticipatory control as a theory of motor preparation: a thalamo-cortical circuit model"](https://doi.org/10.1016/j.neuron.2021.03.009).

Baed on [original version](https://github.com/hennequin-lab/optimal-preparation) 

Please install __PyTorch__ for optimization

## Arm model for straight reach task, and optimize torque
0.reaches: integrate __biomechanics__ into one file 

## Construct target reaches and ISN network
1.soc_construct: same setting as the original

## Find target initial states and readout matrix
2.setup: in Pytorch 

## Feedback preparatory control (instantaneous)
3.vanilla: classical LQR preparation (ignore the noisy trials)

## Full thalamo-cortical loop + preparation strategy
4.1.setup_dynamics: in Pytorch       
4.2.dynamics
