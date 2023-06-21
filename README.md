# PyQuantumControl

Implementing quantum optimal control in Python using `numpy` and `scipy`.

## Overview

This repository provides a framework for simulating and optimizing quantum control pulses using GRAPE and related GOAT-style search scripts. It is aimed at Rydberg atom gates, spin systems, and two-level systems.

See `ARCHITECTURE.md` for the current module layout and execution flow.

## Core Modules

Core logic now lives in packages:

- `core/`: physics, Hamiltonians, time evolution, unitary helpers.
- `control/`: GRAPE/GOAT optimization engines.
- `dataio/`: config loading and HDF5/markdown result writing.
- `viz/`: plotting helpers.
- Legacy root wrappers have been removed; use `core/`, `control/`, `dataio/`, `viz/`, and `scripts/` directly.
- `fidelity.py`, `spinoperators.py`, `oscillators.py`, `rydbergcontrol.py`: shared mathematical utilities and operator builders.

## Repository Structure

- `core/`, `control/`, `dataio/`, `viz/`: shared modules.
- `scripts/`: CLI entry points.
- `configs/`: YAML configuration files for specific strong/weak blockade regimes.
- `analyses/`: Secondary analysis and plotting scripts.
- `bench/`: Benchmarking and verification scripts.

## Usage

### Dependencies
- Python 3
- `numpy`
- `scipy`
- `pyyaml`

### Running Simulations
Run commands from the repository root so local imports resolve correctly.

Example GRAPE run:

```bash
python3 scripts/rydbergcontrol_main.py
```

Example YAML-driven MS gate run:

```bash
python3 scripts/rydberg_ms_gate_main.py --config configs/ms_robust_vanilla.yaml
```

Blockade-regime examples:

```bash
python3 scripts/rydberg_ms_gate_main.py --config configs/ms_1p_strong_blockade.yaml
python3 scripts/rydberg_ms_gate_main.py --config configs/ms_1p_weak_blockade.yaml
python3 scripts/rydberg_ms_gate_main.py --config configs/ms_2p_strong_blockade.yaml
python3 scripts/rydberg_ms_gate_main.py --config configs/ms_2p_weak_blockade.yaml
```

Example manuscript blockade search:

```bash
python3 scripts/optimize_arxiv_spin_echo_blockade_1p.py
```
