# Architecture

## Overview

PyQuantumControl is a modular quantum optimal control codebase built around GRAPE and GOAT-style searches. The repo is now organized as a package-style Python tree with runnable entry points in `scripts/` and shared implementation in `core/`, `control/`, `dataio/`, and `viz/`.

## Layout

- `core/`: physics, Hamiltonians, time evolution, target unitaries, and shared GOAT ramp models.
- `control/`: GRAPE/optimization engines.
- `dataio/`: YAML config loading and HDF5/markdown result writing.
- `viz/`: plotting helpers.
- `scripts/`: CLI entry points.
- `configs/`: problem definitions and regime parameters.
- `analyses/`, `bench/`: secondary analysis and verification scripts.
- Root-level helpers like `params.py`, `fidelity.py`, `spinoperators.py`, `oscillators.py`, `rydbergcontrol.py`, and `controls.py` remain shared support modules.

## Execution Flow

1. A CLI in `scripts/` loads a config or builds a control problem.
2. The script constructs a `PropagatorParameters` or `ControlProblem` instance.
3. `control/optimization.py` runs GRAPE, or a script-specific GOAT routine evaluates a parametric ansatz.
4. `core/evolution.py` propagates the state/unitary under piecewise-constant or IVP dynamics.
5. `core/unitaries.py` and `core/physics.py` provide target gates, ramps, and Hamiltonians.
6. `dataio/io_utils.py` writes results to HDF5 or markdown.
7. `viz/plotting.py` renders pulse figures.

## Control Model

The canonical control abstraction is still dictionary-driven in legacy-style flows:

- `ControlTask`: unitary map or state-to-state map.
- `Initialization`: `Constant`, `Sine`, or `Random`.
- `PropagatorParameters`: time grid, Hamiltonian callback(s), and gradients.
- `HamiltonianParameters`: physical constants for the current protocol.
- `CostFunction` and `CostFunctionGrad`: objective and gradient callables.

The newer package layout keeps that structure, but moves the implementation into reusable modules instead of monolithic scripts.

## Supported Capabilities

- GRAPE optimization for unitary and state objectives.
- GOAT-style optimization with analytic gradients for MS-style ansatzes.
- Piecewise-constant propagation.
- IVP-based propagation for continuous control ansatzes.
- Robust control by averaging over Hamiltonian parameter samples or landmarks.
- Mølmer-Sørensen gate searches in 1-photon and 2-photon blockade regimes.
- Result export to HDF5 and markdown.
- Pulse plotting for paper figures.

## Roadmap

### Near-Term

- Better initialization strategies for GOAT/GRAPE.
- Time-dependent target states.
- Alternate ODE solvers for continuous-control propagation.

### Longer-Term

- Open quantum systems via Lindblad dynamics.
- Hardware pulse-shaping interfaces.
- Parallel robust averaging across CPU cores.

## Notes

This document consolidates the older architecture notes and updates them to the current package layout.
