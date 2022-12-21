# Developer's Guide: The Control Problem Configuration
 
This document describes the dictionary-driven configuration used to define quantum control problems within the PyQuantumControl framework.
 
## The `control_problem` Dictionary
 
Every simulation in a `*_main.py` script centers around a `control_problem` dictionary. This dictionary is passed to `grape.grape()` and orchestrates how the system evolves and how the cost is calculated.
 
### 1. Global Task Settings
- `ControlTask`: Specifies the goal.
    - `'UnitaryMap'`: Optimize for a target unitary matrix.
    - `'StateToStateMap'`: Optimize for a target final state given an initial state.
- `Initialization`: The starting guess for control fields.
    - `'Constant'`, `'Sine'`, or `'Random'`.
 
### 2. Propagator Parameters (`PropagatorParameters`)
Nested dictionary defining the time-discretization:
- `Nsteps`: Number of time steps in the piecewise-constant approximation.
- `Tstep`: Duration of each single time step.
- `Tcontrol`: Total time ($\text{Nsteps} \times \text{Tstep}$).
 
### 3. Hamiltonian Configuration
The framework separates static parameters from those that vary for robust control:
- `HamiltonianParameters`: Dictionary containing physical constants (e.g., Rabi frequencies, detunings). For protocols using external configs, this is typically loaded from a YAML file in `configs/`.
- `HamiltonianUncertainParameters`: (Optional) Defines the range or distribution of parameters for robust optimization.
- `HamiltonianLandmarks`: (Optional) A list of specific parameter sets used to average the cost and gradient for robustness.
 
### 4. Objective and Target
- `CostFunction`: The callable used to compute infidelity (e.g., `objectives.infidelity_unitary`).
- `CostFunctionGrad`: The callable used to compute the gradient of the cost function.
- `UnitaryTarget`: The target unitary matrix $\text{U}_{\text{target}}$ (for `UnitaryMap`).
- `PureStateInitial` / `PureStateTarget`: The initial and target kets (for `StateToStateMap`).
 
## Execution Flow
 
The `grape` engine uses these entries as follows:
1. **Propagator** $\rightarrow$ Reads `PropagatorParameters` to set up the time loop.
2. **Hamiltonian** $\rightarrow$ Uses `HamiltonianParameters` to build the matrices for each step.
3. **Objective** $\rightarrow$ Uses the `CostFunction` and `UnitaryTarget`/`PureStateTarget` to evaluate performance.
4. **Optimization** $\rightarrow$ Iteratively updates control fields using `CostFunctionGrad`.

