# Control paradigms
- Phase control for a two level atom
- Detuning control for a two level atom
- Phase control for Rydberg atoms
- Detuning control for Rydberg atoms
# Time evolution
There are different paradigms for solving the time dependent Schrodinger equation. For each of these paradigms, I need to calculate both the time evolution operator and its gradient with respect to the control variables
- Step wise propagator
- ODE solvers
# Robustness
For robust quantum optimal control, I need the control sequences found to be robust against errors in Hamiltonian parameters.
# Algorithm
The algorithm for robust quantum control is as follows.
## Calculation of the propagator
### Control propagator
This is the unitary time evolution operator implemented during the control
I calculate the propagator and its derivatives simultaneously. This takes the following inputs
- Hamiltonian as a function of time
- Derivatives of the Hamiltonian as a function of time
### Full propagator
This is the unitary time evolution operator implemented by the full protocol
## Calculation of Hamiltonians
I calculate the Hamiltonians and its derivatives simultaneously. This takes the following inputs
- Static parameters of the Hamiltonian
- Controllable parameters of the Hamiltonian
## Calculation of the infidelity cost function
I calculate the infidelity cost function and its derivatives simultaneously. This takes the following inputs.
- Propagators
- Derivatives of the propagators
## Preparation of landmark points
Calculate the weights and the points from the appropriate distribution
