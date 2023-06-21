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
    
    # Dual-control cost function (Omega(t), Delta(t))
    # phi is a flat vector [omega_0...omega_{N-1}, delta_0...delta_{N-1}]
    def cost_function(phi, cp):
        # Split phi into omega and delta
        phi_omega = phi[:N_steps]
        phi_delta = phi[N_steps:]
        
        # Create a sequence of pairs [omega, delta] for the propagator
        phi_pairs = np.stack([phi_omega, phi_delta], axis=1)
        
        from core.evolution import propagator_piecewise
        
        # We redefine the hamiltonian function temporarily to handle the pairs
        original_h_func = cp.propagator_params.hamiltonian_matrix_func
        
        def pair_h_func(phi_pair, h_params):
            return original_h_func(phi_pair, h_params)
            
        cp.propagator_params.hamiltonian_matrix_func = pair_h_func
        U_final = propagator_piecewise(phi_pairs, cp.propagator_params)
        cp.propagator_params.hamiltonian_matrix_func = original_h_func
        
        import fidelity
        return fidelity.unitary_infidelity(U_final, target_u, 9)


    def cost_function_grad(phi, cp):
        # Similar to cost_function, handle dual controls
        phi_omega = phi[:N_steps]
        phi_delta = phi[N_steps:]
        phi_pairs = np.stack([phi_omega, phi_delta], axis=1)
        
        from core.evolution import propagator_gradient_piecewise
        
        original_h_func = cp.propagator_params.hamiltonian_matrix_func
        def pair_h_func(phi_pair, h_params):
            return original_h_func(phi_pair, h_params)
        cp.propagator_params.hamiltonian_matrix_func = pair_h_func
        
        U_final, V_grads_all = propagator_gradient_piecewise(phi_pairs, cp.propagator_params)
        cp.propagator_params.hamiltonian_matrix_func = original_h_func
        
        # V_grads_all is dU/d(phi_pair_n). 
        # For each n, we have dU/d(omega_n) and dU/d(delta_n)
        # Need to combine them with the infidelity gradient.
        
        # This is complex because fidelity_unitary_gradient expects a 1D list of V_k.
        # We can use fidelity.unitary_infidelity_gradient directly.
        import fidelity
        
        grad_phi = np.zeros(2 * N_steps)
        overlap = np.trace(np.dot(target_u.conj().T, U_final))
        
        for n in range(N_steps):
            # V_grads_all[n] is a list [dU/domega_n, dU/ddelta_n]
            # because hamiltonian_ms_gate_grad returns a list of 2 matrices.
            for i in range(2):
                V_kn = V_grads_all[n][i]
                term = np.trace(np.dot(target_u.conj().T, V_kn))
                grad_phi[N_steps * 0 + n if i==0 else N_steps * 1 + n] = (2.0/81.0) * (overlap.conj() * term).real
                
        return grad_phi

    # Build Control Problem
    cp = ControlProblem(
        initialization='Random', # We'll override this because we need 2*Nsteps
        propagator_params=prop_params,
        unitary_target=target_u,
        n_states_unitary=9,
        cost_function=cost_function,
        cost_function_grad=cost_function_grad
    )
    
    # Override phi_initial for dual control
    N_total = 2 * N_steps
    phi_initial = np.random.uniform(-1, 1, N_total)
    
    # We must pass a custom initialization because control.optimization.grape()
    # reads `control_problem.initialization`.
    
    print(f"Starting Dual-Control GRAPE (Omega & Delta) for MS gate...")
    phi_opt, infid_min = grape_dual(cp, phi_initial)
    
    print(f"\nOptimization Complete!")
    print(f"Final Infidelity: {infid_min:.6f}")

def grape_dual(cp, phi_initial):
    import scipy.optimize
    cost_function = cp.cost_function
    cost_function_grad = cp.cost_function_grad
    
    result = scipy.optimize.minimize(
        fun=cost_function, x0=phi_initial, jac=cost_function_grad, method='BFGS', 
        args=(cp,), options={'gtol': 1e-5}
    )
    return result.x, result.fun

if __name__ == "__main__":
    main()
