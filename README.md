# optimal-preparation_Python

python version codes of Optimal anticipatory control as a theory of motor preparation: a thalamo-cortical circuit model

baed on  [original version](https://github.com/hennequin-lab/optimal-preparation) 

Please install __pytorch__ for optimization

## Construct target reaches and ISN network
1.soc_construct: same setting as the original

## Find target initial states and readout matrix
2.setup: use velocity as output for the moment
in Pytorch 

## Feedback preparatory control (instantaneous)
3.vanilla: classical LQR preparation (ignore the noisy trial)

## Full thalamo-cortical loop + preparation strategy
4.1setup_dynamics
4.2dynamics
