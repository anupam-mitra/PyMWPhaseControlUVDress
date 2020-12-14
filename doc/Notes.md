# Representation and computation of a Hamiltonian
## 2018-02-28
Hamiltonian parameters are represented using a dictionary with keys corresponding to parameters of the Hamiltonian. Two different Python functions are used to compute the Hamiltonian matrix and its gradient. For example, for the Rydberg atom problem the Hamiltonian parameters are as follows.
```
    'OmegaRa': Rydberg Rabi frequency of atom a
    'OmegaRb': Rydberg Rabi frequency of atom b
    'DeltaRa': Rydberg detuning of atom a
    'DeltaRb': Rydberg detuning of atom b
    'OmegaMWa': Microwave Rabi frequency of atom a
    'OmegaMWb': Microwave Rabi frequency of atom b
    'DeltaMWa': Microwave detuning of atom a
    'DeltaMWb': Microwave detuning of atom b
```

This dictionary is passed as the argument `h_params` to the function `rydbergatoms.hamiltonian_PerfectBlockade` which calculates the Hamiltonian matrix and to the function `rydbergatoms.hamiltonian_grad_PerfectBlockade` which calcuates the derivative of the Hamiltonian matrix with respect to the phase of the microwave. Both these functions also take an argument `phi`, which is the phase of the microwave.

## 2018-09-19
The Hamiltonian and its gradient can also be computed in the same Python function. For example, for phase control of a two level atom, the function `twolevelsystem.calc_hamiltonian_phase_control` calculates the Hamiltonian for given values of the phase, detuning and Rabi frequency. It also calculates the first derivative and second derivative of the Hamiltonian with respect to the phase of the electromagnetic wave.

For detuning control, the dependence of the Hamiltonian as terms of the form `cos(detuning * t)`. This had explicit time dependence. Therefore, the propagator computation requires an explicit time dependence.

# Representation and computation of propagators
## 2017-07-15
Propagators, or time, evolution operators are computed using a piece wise constant time dependence of the Hamiltonian. "Piece" and "step" are used interchangeably to refer to a duration of time for which the Hamiltonian is kept constant. The Hamiltonian is kept constant for some time `Tstep`. This is repeated for `Nsteps` steps. The total control time is `Tcontrol`

```
    'HamiltonianParameters' : \
	Dictionary containing the parameters of the Hamiltonian, \
    'HamiltonianMatrix' : \
	Python function which calculates the Hamiltonian matrix for a given \
	value of the microwave phase, \
    'HamiltonianMatrixGradient' : \
	Python function which calculates the derivative of the Hamiltonian \
	matrix for a given value of the microwave phase, \
    'Nsteps' : Number of steps of keeping the Hamiltonian constant, \
    'Tstep' : Time duration of each step, \
    'Tcontrol' :  Nsteps * Tstep, \
```

# Representation and computation of cost functions and control problem
## 2018-02-28
Cost functions which are minimized correspond to infidelity of the control task. So far only infidelity is considered. For a gradient based search for a sequence of phase steps, the gradient of the infidelity is also computed. For implementing a pure state map, the cost function is `costfunctions.infidelity` and the gradient of the cost function is `costfunctions.infidelity_gradient`. For implementing a unitary map, the cost function is `costfunctions.infidelity_unitary` and the gradient is `costfunctions.infidelity_unitary_gradient`. 

Robust control problems are implemented by averaging the cost function over landmark points. For implementing a robust pure state map, the cost function is `robustcostfunctions.infidelity` and the gradient of the cost function is `robustcostfunctions.infidelity_gradient`. For implementing a unitary map, the cost function is `robustcostfunctions.infidelity_unitary` and the gradient is `robustcostfunctions.infidelity_unitary_gradient`. 

A control problem is represented by parameters in a dictionary. `ControlTask` specifies whether the control is to prepare a unitary map (`UnitaryMap`) or a pure state map (`StateToStateMap`). The initial guess for the control sequence is determined by `Initialization`. At present `Constant`, `Sine` and `Random` are implemented.

An example of a pure state map is the following.
```
    'ControlTask' : 'StateToStateMap', \
    'Initialization' : 'Constant', \
    'PropagatorParameters': propagator_parameters, \
    'PureStateInitial': rydbergatoms.ket_00, \
    'PureStateTarget': calc_targetdressedstate(hamiltonian_parameters), \
    #'PureStateTarget': (rydbergatoms.ket_01 + rydbergatoms.ket_10)/sqrt(2), \
    'CostFunction' : robustcostfunctions.infidelity, \
    'CostFunctionGrad' : robustcostfunctions.infidelity_gradient, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianUncertainParameters' : hamiltonian_uncertain_parameters, \

```

## 2018-03-28
For robust unitary maps including adiabatic dressing before control and adiabatic undressing after control, the time evolution before the control and the time evolution after the control also need to be considered. For this, the cost function is `robustadiabaticcostfunctions.infidelity_unitary`.

## 2018-06-05
The computation of the infidelity and its gradient are done simultaneously for robust unitary maps to be implemented with adiabatic dressing and undress. This is done in the function `robustadiabaticcostfunctions.infidelity_unitary`. This also requires calculation of the time evolution operator and its gradient to be done simultaneously. The mean of the cost function and its gradient over some landmark points is minimized. The landmakrk points are parameterized by the parameter `HamiltonianLandmarks`

An example of a unitary map is the following.
```
    'ControlTask' : 'UnitaryMap', \
    'Initialization' : 'Sine', \
    'UnitaryTarget': u_target, \
    'PropagatorParameters': propagator_parameters, \
    'CostFunction' : robustadiabaticcostfunctions.infidelity_unitary, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianLandmarks': hamiltonian_landmarks_list,
```

