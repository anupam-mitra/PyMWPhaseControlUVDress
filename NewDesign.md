# System Design Specification
 
## Architecture Overview
 
The repository implements a modular framework for Quantum Optimal Control using the GRAPE (Gradient Ascent Pulse Engineering) algorithm. The system is designed as a flat Python structure where core physics and optimization logic are decoupled from specific control problem configurations.
 
## Core Design Components
 
### 1. Optimization Engine (`grape.py`)
The central orchestrator of the control problem. It uses a dictionary-driven approach:
- **Control Problem**: A configuration dictionary containing Hamiltonian parameters, propagator settings, and the cost function.
- **Iteration Loop**: Manages the optimization process, updating control fields based on gradients provided by the propagator and objective functions.
- **API**: Standardized to handle `CostFunctionGrad` types to ensure consistency across different objective functions.
 
### 2. Time Evolution and Propagation (`timeevolution.py`)
Implements the time-dependent Schrödinger equation using a piecewise-constant approximation:
- **Piecewise Matrix Exponential**: The total evolution is computed as a product of exponentials over $N$ time steps.
- **Gradient Calculation**: Efficiently computes the gradient of the unitary with respect to control fields by propagating the state forward and the target backward.
- **Unified Interface**: Provides a single entry point for both unitary propagation and gradient evaluation.
 
### 3. Objective Functions (`objectives.py`)
Defines the goals of the optimization:
- **Infidelity Metrics**: Implementations of state-to-state and unitary-to-unitary fidelity.
- **Robust Control**: Implements cost averaging over a distribution of Hamiltonian parameters (e.g., detuning offsets) to create pulses that are insensitive to experimental noise.
- **Robust Gradients**: Calculates the average gradient across the sampled parameter space.
 
### 4. Hamiltonian Modules
Physical models are isolated into dedicated modules to ensure reusability:
- `rydberghamiltonians.py`: Models for Rydberg atoms, including dressing and UV control.
- `rydberg_ms_hamiltonians.py`: Specific Hamiltonian models for the Mølmer-Sørensen gate, supporting both 1-photon and 2-photon regimes.
- `twoplevelhamiltonians.py`: Basic two-level system dynamics.
- `spinhamiltonians.py`: Spin-chain and multi-qubit interaction models.
 
### 5. Mathematical Utilities
Shared helpers for quantum mechanics operations:
- `fidelity.py`: General fidelity and distance measures.
- `spinoperators.py`: Pauli matrices and spin-operator builders.
- `oscillators.py`: Harmonic oscillator basis and operators.
- `rydbergcontrol.py`: Specific helper functions for Rydberg state transitions.
 
## Control Flow
 
1. **Configuration**: 
    - A `*_main.py` script defines the `control_problem` dictionary.
    - For complex protocols, physical parameters and regime settings are loaded from YAML files in `configs/`.
2. **Initialization**: Control fields are initialized (Random or Constant).
3. **Optimization**: `grape.grape()` is called, which iteratively:
    - Calls `timeevolution.py` to evolve the system.
    - Calls `objectives.py` to evaluate the cost and gradient.
    - Updates control fields to minimize the cost.
4. **Analysis**: Resulting waveforms are analyzed using scripts in `analyses/` or `bench/`.

