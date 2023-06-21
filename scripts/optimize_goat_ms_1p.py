import numpy as np
import scipy.optimize
from core.evolution import propagator_with_gradient_ivp
from params import PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.unitaries import get_cz_target
import core.ms_goat_models as mgm
from core.ms_goat_models import one_photon_ramp_val, one_photon_ramp_grad

CURRENT_TCONTROL = 20.0

def adiabatic_ramp_1p_val(t, params):
    return one_photon_ramp_val(t, params)


def adiabatic_ramp_1p_grad(t, params):
    return one_photon_ramp_grad(t, params)

# Global for the ansatz function
CURRENT_PARAMS = [628.318, 1.841, 5.977]

def phi_ansatz(t, h_params):
    return adiabatic_ramp_1p_val(t - CURRENT_TCONTROL / 2.0, CURRENT_PARAMS)

def goat_hamiltonian(phi_val, h_params):
    return hamiltonian_ms_gate(phi_val, h_params)

def goat_hamiltonian_grad(t, phi_val, h_params, params_vec):
    # H = H(phi(t, alpha), h_params)
    # dH/dalpha_k = (dH/dphi) * (dphi/dalpha_k)
    dH_dphi = hamiltonian_ms_gate_grad(phi_val, h_params)
    dphi_dalpha = adiabatic_ramp_1p_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    
    res_grads = []
    for k in range(len(params_vec)):
        res_grads.append(dphi_dalpha[k] * dH_dphi)
    return res_grads

def objective_function(params, target_u, prop_params):
    global CURRENT_PARAMS, CURRENT_TCONTROL
    CURRENT_PARAMS = params
    CURRENT_TCONTROL = prop_params.Tcontrol
    mgm.CURRENT_TCONTROL = CURRENT_TCONTROL
    
    try:
        U, V_grads = propagator_with_gradient_ivp(phi_ansatz, prop_params, params)
        
        if np.any(np.isnan(U)) or np.any(np.isinf(U)):
            return 1.0, np.zeros(len(params))
            
        overlap = np.trace(np.dot(target_u.conj().T, U))
        fidelity = (1.0/81.0) * np.abs(overlap)**2
        fidelity = np.clip(fidelity, 0.0, 1.0)
        
        grad_params = [(2.0/81.0) * (overlap.conj() * np.trace(np.dot(target_u.conj().T, V_k))).real for V_k in V_grads]
        return 1.0 - fidelity, -np.array(grad_params)
    except:
        return 1.0, np.zeros(len(params))

if __name__ == "__main__":
    # Fixed Delta_max for 1-photon case
    h_params = {
        'regime': '1-photon',
        'V_rr': 0.628,
        'Delta_max': 0.1
    }
    
    prop_params = PropagatorParameters(
        Nsteps=1024,
        Tstep=20.0/1024,
        Tcontrol=20.0,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )
    
    target_u = get_cz_target()
    
    # [O_max, tw, tc]
    initial_guess = [628.318, 1.841, 5.977]
    bounds = [
        (10.0, 2000.0), # O_max
        (0.1, 10.0),    # tw
        (0.1, 20.0),    # tc
    ]
    
    print("Evaluating initial infidelity for 1-photon CZ...")
    infid, _ = objective_function(initial_guess, target_u, prop_params)
    print(f"Initial Infidelity: {infid:.6f}")
    
    if infid > 1e-4:
        print("Starting GOAT optimization...")
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
