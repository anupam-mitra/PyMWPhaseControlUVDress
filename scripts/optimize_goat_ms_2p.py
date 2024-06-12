import numpy as np
import scipy.optimize
from core.evolution import propagator_with_gradient_ivp
from core.params import PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.unitaries import get_cz_target
import core.ms_goat_models as mgm
from core.ms_goat_models import two_photon_ramp_val, two_photon_ramp_grad

CURRENT_TCONTROL = 20.0

def adiabatic_ramp_2p_val(t, params):
    return two_photon_ramp_val(t, params)


def adiabatic_ramp_2p_grad(t, params):
    return two_photon_ramp_grad(t, params)

# Global for the ansatz function
CURRENT_PARAMS = [628.318, 0.1, 0.1, 1.841, 5.977]

def phi_ansatz(t, h_params):
    return adiabatic_ramp_2p_val(t - CURRENT_TCONTROL / 2.0, CURRENT_PARAMS)

def goat_hamiltonian(phi_val, h_params):
    return hamiltonian_ms_gate(phi_val, h_params)

def goat_hamiltonian_grad(t, phi_val, h_params, params_vec):
    # The GOAT system needs dH/dalpha_k
    # H = H(omega(t, alpha), delta(t, alpha), h_params)
    # dH/dalpha_k = (dH/domega) * (domega/dalpha_k) + (dH/ddelta) * (ddelta/dalpha_k)
    
    dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, h_params)
    
    # Get la-gradients of controls w.r.t params
    dphi_dalpha = adiabatic_ramp_2p_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    
    res_grads = []
    for k in range(len(params_vec)):
        # la-gradient: (dH/domega * domega/dalpha_k) + (dH/ddelta * ddelta/dalpha_k)
        res_grads.append(dH_domega * dphi_dalpha[0, k] + dH_ddelta * dphi_dalpha[1, k])
    return res_grads

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
    import yaml

    # Load YAML
    with open("configs/ms_2p_weak_blockade.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    h_params = config['hamiltonian_params']
    # Ensure we have a Delta_max for the optimization
    if 'Delta_max' not in h_params:
        # If not in YAML, use a default or extract from regime logic
        h_params['Delta_max'] = 0.1 
    
    prop_params = PropagatorParameters(
        Nsteps=1024,
        Tstep=20.0 / 1024,
        Tcontrol=20.0,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )
    
    target_u = get_cz_target()
    
    # Initial parameters for optimization: [O_max, D_max, D_min, tw, tc]
    initial_guess = [
        h_params['Omega_1a_max'],
        0.1,  # D_max default
        0.1,  # D_min default
        h_params['t_gaussian_width'],
        h_params['t_constant_duration'],
    ]
    
    bounds = [
        (10.0, 2000.0), # O_max
        (-10.0, 10.0),   # D_max
        (-10.0, 10.0),   # D_min
        (0.1, 10.0),      # tw
        (0.1, 20.0),      # tc
    ]
    
    print("Evaluating fidelity of YAML parameters...")
    # To evaluate YAML params, we need to match the initial_guess structure
    # YAML doesn't have D_min, so we'll use a reasonable guess or D_max.
    infid, _ = objective_function(initial_guess, target_u, prop_params)
    print(f"Initial Infidelity: {infid:.6f}")
    
    if infid > 1e-4:
        print("Starting GOAT optimization for CZ gate (Vectorized Control)...")
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
