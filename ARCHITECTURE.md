# Architecture

## Overview

PyQuantumControl is a quantum optimal control codebase built around GRAPE and GOAT-style searches. The repository now uses a shared-package layout for reusable physics and optimization code, with script-driven workflows for experiments, benchmarks, and manuscript figures.

The current codebase is a hybrid of newer dataclass-based modules and older dictionary-based entry points. Both are still supported in a few places.

## Layout

- `core/`: Hamiltonians, time evolution, fidelities, target unitaries, pulse ansatz models, utility math, and control parameter dataclasses.
- `control/`: GRAPE wrapper, objective functions, and Rydberg-specific control-problem builders.
- `dataio/`: YAML loading and HDF5/Markdown result serialization.
- `viz/`: plotting helpers and figure generation.
- `scripts/`: runnable CLIs for optimization, evaluation, analysis, and paper figures.
- `configs/`: YAML problem definitions for MS gate regimes.
- `analyses/`, `bench/`: secondary analysis and verification scripts.

## Core Abstractions

- `PropagatorParameters`: time grid plus Hamiltonian callbacks and parameters.
- `ControlProblem`: optimization container holding the propagator, objective, targets, and optional robust-control metadata.
- `ControlProblem.from_dict(...)`: bridge for legacy dictionary-style scripts.
- `control.objectives`: infidelity and gradient functions for state, unitary, and robust objectives.

## Execution Paths

### GRAPE

1. A script builds a legacy dict or `ControlProblem` instance.
2. `control.optimization.grape()` normalizes legacy input and calls `scipy.optimize.minimize`.
3. `control.objectives` evaluates the objective and gradient.
4. `core.evolution` propagates piecewise-constant controls and gradients.
5. `core.unitaries` and `core.physics` provide targets and Hamiltonians.

Typical entry points:

- `scripts/rydbergcontrol_main.py`
- `scripts/rydbergcontrol_unitary_main.py`
- `scripts/rydbergrobustcontrol_main.py`
- `scripts/rydbergrobustcontrol_unitary_main.py`
- `scripts/optimize_grape_ms_cz.py`
- `scripts/rydberg_ms_gate_main.py`

### GOAT / IVP

1. A script defines a parametric ansatz for the control waveform.
2. `core.evolution.propagator_with_gradient_ivp()` integrates the dynamics and parameter sensitivities.
3. The script converts the final unitary and gradients into a scalar infidelity objective.
4. `scipy.optimize.minimize` updates waveform parameters.

Typical entry points:

- `scripts/optimize_goat_ms.py`
- `scripts/optimize_goat_ms_1p.py`
- `scripts/optimize_goat_ms_2p.py`
- `scripts/optimize_goat_ms_2p_random.py`
- `scripts/optimize_goat_ms_gaussian_cosine.py`
- `scripts/optimize_ms_gate_ivp.py`

### Analysis and Figures

- `scripts/evaluate_ms_fidelity.py`: evaluate candidate MS pulses.
- `scripts/analyze_hamiltonian_spectrum.py`: inspect spectra.
- `scripts/plot_arxiv_paper_pulses.py`: generate manuscript pulse figures.
- `analyses/` and `bench/`: ad hoc diagnostics and validation.

## Numerical Layers

- `core/evolution.py`: piecewise-constant propagation, IVP propagation, and gradient propagation.
- `core/physics.py`: 9D MS gate Hamiltonian and gradients.
- `core/rydberghamiltonians.py`: legacy perfect-blockade Rydberg Hamiltonians.
- `core/ms_goat_models.py`: one-photon and two-photon GOAT ramps plus Gaussian-cosine ansatzes.
- `core/unitaries.py`: target gates, spin-echo composition, and ramp evaluation helpers.

## Data Flow

1. Scripts load YAML config or construct parameters inline.
2. Waveform and Hamiltonian parameters are mapped into a propagator description.
3. Dynamics are evaluated with either piecewise-constant propagation or IVP integration.
4. Objectives return infidelity and gradients to the optimizer.
5. `dataio/io_utils.py` writes HDF5 and Markdown outputs.
6. `viz/plotting.py` renders waveform figures for reports and papers.

## Notes

- The repo no longer relies on root-level helper modules for core functionality.
- Some scripts still duplicate setup logic instead of sharing a single orchestration layer.
- `configs/` is the canonical location for YAML problem definitions.
- `dataio/io_utils.py` and `viz/plotting.py` are primarily support layers, not the main control engine.
