import numpy as np

from dataio.io_utils import load_yaml
from params import PropagatorParameters
from core.physics import hamiltonian_ms_gate
from core.evolution import propagator_ivp
from core.unitaries import get_ms_yy_target, one_photon_ramp


def main():
    config = load_yaml('configs/ms_robust_vanilla.yaml')
    h_params = config['hamiltonian_params']
    c_params = config['control_params']

    prop_params = PropagatorParameters(
        Nsteps=c_params['Nsteps'],
        Tstep=c_params['Tcontrol'] / c_params['Nsteps'],
        Tcontrol=c_params['Tcontrol'],
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None
    )
    
    # Construct phi sequence (tuples of Omega, Delta)
    t_grid = np.linspace(-prop_params.Tcontrol / 2, prop_params.Tcontrol / 2, prop_params.Nsteps)
    phi = [one_photon_ramp(t, {**h_params, **c_params, 'Tcontrol': prop_params.Tcontrol, 'Nsteps': prop_params.Nsteps}) for t in t_grid]
    
    print(f"Evaluating fidelity for ms_robust_vanilla.yaml...")
    U = propagator_ivp(phi, prop_params)
    
    target_u = get_ms_yy_target()
    overlap = np.trace(np.dot(target_u.conj().T, U))
    fidelity = (1.0/81.0) * np.abs(overlap)**2
    
    print(f"Calculated Fidelity: {fidelity:.6f}")
    print(f"Infidelity: {1.0 - fidelity:.6f}")

if __name__ == "__main__":
    main()
