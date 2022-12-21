# PyQuantumControl
 
Implementing quantum optimal control in Python using `scipy` and `numpy`.
 
## Overview
 
This repository provides a framework for simulating and optimizing quantum control pulses using the GRAPE (Gradient Ascent Pulse Engineering) algorithm. It is designed for research into Rydberg atoms, spin systems, and two-level systems.
 
## Core Modules
 
The repository is structured with core logic at the root to facilitate easy imports across scripts:
 
- `grape.py`: The core optimization engine implementing the GRAPE algorithm.
- `timeevolution.py`: Handles propagation, time-evolution, and gradient calculations for the quantum state/unitary.
- `objectives.py`: Contains various cost functions, infidelity metrics, and robust averaging logic.
- `rydberghamiltonians.py`, `twoplevelhamiltonians.py`, `spinhamiltonians.py`: Define the physical Hamiltonian models.
- `rydberg_ms_hamiltonians.py`: Hamiltonian and gradient for the Mølmer-Sørensen (MS) gate protocol.
- `fidelity.py`, `spinoperators.py`, `oscillators.py`, `rydbergcontrol.py`: Shared mathematical utilities and operator builders.
 
## Repository Structure
 
- Root Directory: Contains core modules and `*_main.py` entry points for specific control problems.
- `configs/`: YAML configuration files for different physical regimes and protocols.
- `analyses/`: Scripts for analyzing optimization results and performing secondary simulations.
- `bench/`: Benchmarking scripts for performance and accuracy verification.
 
## Usage
 
### Dependencies
- Python 3
- `numpy`
- `scipy`
- `pyyaml`
 
### Running Simulations
The project uses a dictionary-driven configuration. To run a simulation, execute the relevant `*_main.py` script from the repository root:
 
```bash
python3 rydbergcontrol_main.py
```
 
For protocols using YAML configurations (e.g., the MS gate), specify the configuration file:
 
```bash
python3 rydberg_ms_gate_main.py --config configs/ms_robust_vanilla.yaml
```
 
Note: Ensure you run commands from the root directory so that local imports resolve correctly.

