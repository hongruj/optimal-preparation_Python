# optimal-preparation_Python

python version codes of Optimal anticipatory control as a theory of motor preparation: a thalamo-cortical circuit model

baed on  [original version](https://github.com/hennequin-lab/optimal-preparation) with some changes and deletions

## update!
There is a problem in output_lqr, where the cost value turn to negative

method 1: gradient_output_lqr, Adam in numpy
method 2: minimize_output_lqr, scipy.optimize.minimize (very slow with large cost)



## Construct target reaches and ISN network
1.soc_construct: same setting as the original

## Find target initial states and readout matrix
2.setup: use velocity as output for the moment
in Pytorch 

## Move-phase simulation
3.1move_phase(2014 eigenvectors): use top 6 eigenvectors as initial states   
3.2move_phase(xstars): optimized initial states from setup 

## Feedback preparatory control
4.vanilla: classical LQR preparation (with many changes in lib)
