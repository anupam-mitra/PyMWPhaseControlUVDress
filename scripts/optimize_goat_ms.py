import numpy as np
import scipy.optimize
from core.evolution import propagator_with_gradient_ivp
from core.params import PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.unitaries import get_ms_yy_target
import core.ms_goat_models as mgm
from core.ms_goat_models import one_photon_ramp_val, one_photon_ramp_grad

CURRENT_TCONTROL = 8.0703458766310

def adiabatic_ramp_val(t, params):
    return one_photon_ramp_val(t, params)


def adiabatic_ramp_grad(t, params):
    return one_photon_ramp_grad(t, params)

def goat_hamiltonian(phi_val, h_params):
    return hamiltonian_ms_gate(phi_val, h_params)

def goat_hamiltonian_grad(t, phi_val, h_params, params_vec):
    H_grad_omega, H_grad_delta = hamiltonian_ms_gate_grad(phi_val, h_params)
    phi_grads = adiabatic_ramp_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    res_grads = []
    for k in range(len(params_vec)):
        H_k = phi_grads[0, k] * H_grad_omega + phi_grads[1, k] * H_grad_delta
        res_grads.append(H_k)
    return res_grads

# Global for the ansatz function
CURRENT_PARAMS = [1.0, -5.0, -0.16, 1.909859, 0.430928]

def phi_ansatz(t, h_params):
    return adiabatic_ramp_val(t - CURRENT_TCONTROL / 2.0, CURRENT_PARAMS)

def objective_function(params, target_u, prop_params):
    global CURRENT_PARAMS, CURRENT_TCONTROL
    CURRENT_PARAMS = params
    CURRENT_TCONTROL = prop_params.Tcontrol
    mgm.CURRENT_TCONTROL = CURRENT_TCONTROL
    
    U, V_grads = propagator_with_gradient_ivp(phi_ansatz, prop_params, params)
    
    overlap = np.trace(np.dot(target_u.conj().T, U))
    fidelity = (1.0/81.0) * np.abs(overlap)**2
    
    grad_params = []
    for V_k in V_grads:
        term = np.trace(np.dot(target_u.conj().T, V_k))
        grad_params.append((2.0/81.0) * (overlap.conj() * term).real)
        
    return 1.0 - fidelity, -np.array(grad_params)

if __name__ == "__main__":
    h_params = {'V_rr': 10.0}
    prop_params = PropagatorParameters(
        Nsteps=100,
        Tstep=8.0703458766310 / 100,
        Tcontrol=8.0703458766310,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )
    
    target_u = get_ms_yy_target()
    # [Omega_max, Delta_max, Delta_min, tw, tc]
    initial_guess = [1.0, -5.0, -0.16, 1.909859, 0.430928]
    bounds = [(0.1, 10.0), (-20.0, 0.0), (-1.0, 0.0), (0.1, 5.0), (0.01, 2.0)]
    
    print("Evaluating la-fidelity of la-paper parameters...")
    infid, _ = objective_function(initial_guess, target_u, prop_params)
    print(f"Initial Infidelity: {infid:.6f}")
    
    if infid > 1e-4:
        print("Starting GOAT optimization to reach < 1e-4...")
        res = scipy.optimize.minimize(
            objective_function, 
            initial_guess, 
            args=(target_u, prop_params),
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-8}
        )
        print("Optimization Result:\n", res)
        print("Optimal Parameters:", res.x)
        print("Final Infidelity:", res.fun)
    else:
        print("Fidelity is already sufficient.")
