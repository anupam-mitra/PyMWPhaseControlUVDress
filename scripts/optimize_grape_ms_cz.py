import numpy as np
import scipy.optimize
from control.optimization import grape
from params import ControlProblem, PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.unitaries import get_ms_yy_target
from objectives import infidelity_unitary, infidelity_unitary_gradient

def main():
    # Hyperparameters
    V_rr = 1.0
    T_control = 2.0
    N_steps = 50
    
    # Hamiltonian parameters
    h_params = {
        'regime': '1-photon',
        'V_rr': V_rr,
        'Delta_max': -1.0, # Constant detuning
    }
    
    # Propagator Parameters
    prop_params = PropagatorParameters(
        Nsteps=N_steps,
        Tstep=T_control / N_steps,
        Tcontrol=T_control,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=hamiltonian_ms_gate_grad
    )
    
    # Target Unitary (MS_yy)
    target_u = get_ms_yy_target()
    
    # Define cost function for GRAPE
    def cost_function(phi, cp):
        return infidelity_unitary(phi, cp)

    def cost_function_grad(phi, cp):
        return infidelity_unitary_gradient(phi, cp)

    # Build Control Problem
    cp = ControlProblem(
        initialization='Random',
        propagator_params=prop_params,
        unitary_target=target_u,
        n_states_unitary=9,
        cost_function=cost_function,
        cost_function_grad=cost_function_grad
    )
    
    print(f"Starting GRAPE optimization for MS gate (T={T_control}, N={N_steps})...")
    phi_opt, infid_min = grape(cp)
    
    print(f"\nOptimization Complete!")
    print(f"Final Infidelity: {infid_min:.6f}")

if __name__ == "__main__":
    main()
