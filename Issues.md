# Capabilities & Roadmap

## Current Capabilities

The PyQuantumControl framework currently supports the following quantum control paradigms and features:

### Control Paradigms
- **Two-Level Systems**: Optimization of phase and detuning controls for basic qubit transitions.
- **Rydberg Atoms**: Control of Rydberg state transitions, including complex dressing and undressing protocols.
- **Spin Systems**: Control of multi-qubit interactions and spin-chain dynamics.

### Technical Features
- **GRAPE Algorithm**: Full implementation of Gradient Ascent Pulse Engineering.
- **Robust Control**: Ability to find control pulses that are insensitive to parameter fluctuations by averaging costs and gradients over landmark points.
- **Piecewise-Constant Propagation**: Efficient time-evolution using matrix exponentials over discrete time steps.
- **Flexible Objectives**: Support for both state-to-state and unitary-to-unitary infidelity metrics.

## System Roadmap

### Short-Term Goals
- [ ] Implement advanced initialization strategies (e.g., based on adiabatic pulses).
- [ ] Add support for time-dependent target states.
- [ ] Integrate higher-order ODE solvers as an alternative to piecewise-constant propagation.

### Long-Term Goals
- [ ] Support for open quantum systems (Lindblad master equation).
- [ ] Integration with external pulse-shaping hardware interfaces.
- [ ] Parallelization of robust cost averaging across multiple CPU cores.
