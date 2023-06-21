import numpy as np
import control.optimization as grape
import objectives
import core.physics as rydberg_ms_hamiltonians
import argparse
import os

def get_ms_yy_target():
    """
    Returns the target unitary for the MS_yy gate in the 9D basis.
    Basis: {|0,0>, |0,1>, |0,r>, |1,0>, |1,1>, |1,r>, |r,0>, |r,1>, |r,r>}
    The MS_yy gate acts on the subspace {|0,0>, |0,1>, |1,0>, |1,1>}
    """
    # Qubit subspace indices: 0, 1, 3, 4
    # MS_yy = exp(-i * pi/4 * sigma_y \otimes sigma_y)
    # sigma_y \otimes sigma_y in the qubit basis:
    # [ 0  0  0 -1]
    # [ 0  0  1  0]
    # [ 0  1  0  0]
    # [-1  0  0  0]
    
    U_qubit = np.cos(np.pi/4) * np.eye(4) - 1j * np.sin(np.pi/4) * np.array([
        [ 0, 0, 0, -1],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [-1, 0, 0, 0]
    ])
    
    U_full = np.eye(9, dtype=complex)
    qubit_indices = [0, 1, 3, 4]
    
    for i in range(4):
        for j in range(4):
            U_full[qubit_indices[i], qubit_indices[j]] = U_qubit[i, j]
            
    return U_full

def main():
    import yaml

    parser = argparse.ArgumentParser(description="Run MS gate GRAPE optimization with a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    h_params = config['hamiltonian_params']
    c_params = config['control_params']

    # Propagator parameters
    Nsteps = c_params['Nsteps']
    Tcontrol = c_params['Tcontrol']
    Tstep = Tcontrol / Nsteps
    
    propagator_params = {
        'HamiltonianParameters': h_params,
        'HamiltonianMatrix': rydberg_ms_hamiltonians.hamiltonian_ms_gate,
        'HamiltonianMatrixGradient': rydberg_ms_hamiltonians.hamiltonian_ms_gate_grad,
        'Nsteps': Nsteps,
        'Tstep': Tstep,
        'Tcontrol': Tcontrol,
    }
    
    # Control Problem
    control_problem = {
        'ControlTask': 'UnitaryMap',
        'Initialization': c_params['Initialization'],
        'UnitaryTarget': get_ms_yy_target(),
        'PropagatorParameters': propagator_params,
        'CostFunction': objectives.infidelity_unitary,
        'CostFunctionGrad': objectives.infidelity_unitary_gradient,
    }
    
    print(f"Starting GRAPE optimization using config {args.config}...")
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
    
    print(f"Optimization Complete.")
    print(f"Minimum Infidelity: {infidelity_min}")
    print(f"Optimal Control Fields: {phi_opt}")

if __name__ == '__main__':
    main()
