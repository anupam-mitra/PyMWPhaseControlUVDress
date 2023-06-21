import numpy as np
import scipy.optimize
import random
from core.evolution import propagator_with_gradient_ivp
from params import PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.unitaries import get_cz_target
import core.ms_goat_models as mgm
from core.ms_goat_models import two_photon_ramp_val, two_photon_ramp_grad

CURRENT_TCONTROL = 20.0

def adiabatic_ramp_2p_val(t, params):
    O_max, D_max, D_min, tw, tc = params
    T_total = CURRENT_TCONTROL
    t2, t3 = -tc/2.0, tc/2.0
    t1, t4 = -T_total/2.0, T_total/2.0
    if t < t2:
        omega = O_max * np.exp(-(t - t2)**2 / (2 * tw**2))
    elif t <= t3:
        omega = O_max
    else:
        omega = O_max * np.exp(-(t - t3)**2 / (2 * tw**2))
    if t < t2:
        delta = D_max + (D_min - D_max) * (t - t1) / (t2 - t1) if (t2-t1) != 0 else D_max
    elif t <= t3:
        delta = D_min
    else:
        delta = D_min + (D_max - D_min) * (t - t3) / (t4 - t3) if (t4-t3) != 0 else D_min
    return np.array([omega, delta])

def adiabatic_ramp_2p_grad(t, params):
    O_max, D_max, D_min, tw, tc = params
    T_total = CURRENT_TCONTROL
    t2, t3 = -tc/2.0, tc/2.0
    t1, t4 = -T_total/2.0, T_total/2.0
    grad = np.zeros((2, 5)) 
    if t < t2:
        val = np.exp(-(t - t2)**2 / (2 * tw**2))
        grad[0, 0], grad[0, 3] = val, O_max * val * (t - t2)**2 / (tw**3)
        grad[0, 4] = -O_max * val * (t - t2) / (2.0 * tw**2)
    elif t <= t3:
        grad[0, 0] = 1.0
    else:
        val = np.exp(-(t - t3)**2 / (2 * tw**2))
        grad[0, 0], grad[0, 3] = val, O_max * val * (t - t3)**2 / (tw**3)
        grad[0, 4] = O_max * val * (t - t3) / (2.0 * tw**2)
    if t < t2:
        denom = (t2 - t1)
        if denom != 0:
            frac = (t - t1) / denom
            grad[1, 1], grad[1, 2] = 1.0 - frac, frac
            grad[1, 4] = 2.0 * (D_min - D_max) * (t - t1) / (T_total - tc)**2
    elif t <= t3:
        grad[1, 2] = 1.0
    else:
        denom = (t4 - t3)
        if denom != 0:
            frac = (t - t3) / denom
            grad[1, 1], grad[1, 2] = frac, 1.0 - frac
            grad[1, 4] = 2.0 * (D_max - D_min) * (t - t4) / (T_total - tc)**2
    return grad

CURRENT_PARAMS = [628.318, 0.1, 0.1, 1.841, 5.977]
def phi_ansatz(t, h_params): return adiabatic_ramp_2p_val(t - CURRENT_TCONTROL / 2.0, CURRENT_PARAMS)

def goat_hamiltonian(phi_val, h_params):
    return hamiltonian_ms_gate(phi_val, h_params)

def goat_hamiltonian_grad(t, phi_val, h_params, params_vec):
    dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, h_params)
    dphi_dalpha = adiabatic_ramp_2p_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    return [dH_domega * dphi_dalpha[0, k] + dH_ddelta * dphi_dalpha[1, k] for k in range(len(params_vec))]

def objective_function(params, target_u, prop_params):
    global CURRENT_PARAMS, CURRENT_TCONTROL
    CURRENT_PARAMS = params
    CURRENT_TCONTROL = prop_params.Tcontrol
    mgm.CURRENT_TCONTROL = CURRENT_TCONTROL
    
    try:
        U, V_grads = propagator_with_gradient_ivp(phi_ansatz, prop_params, params)
        
        # Unitarity Check: if the matrix has exploded or vanished, return max infidelity
        if np.any(np.isnan(U)) or np.any(np.isinf(U)):
            return 1.0, np.zeros(len(params))
        
        # Basic check: U * U^dagger approx I
        if not np.allclose(np.dot(U, U.conj().T), np.eye(9), atol=1e-2):
            # If not unitary, penalize based on deviation from unitarity
            # This guides the optimizer away from unstable regions
            unit_err = np.linalg.norm(np.dot(U, U.conj().T) - np.eye(9))
            return 1.0 + unit_err, np.zeros(len(params))

        overlap = np.trace(np.dot(target_u.conj().T, U))
        fidelity = (1.0/81.0) * np.abs(overlap)**2
        
        # Ensure fidelity is clipped to [0, 1] to avoid numerical noise
        fidelity = np.clip(fidelity, 0.0, 1.0)
        
        grad_params = [(2.0/81.0) * (overlap.conj() * np.trace(np.dot(target_u.conj().T, V_k))).real for V_k in V_grads]
        return 1.0 - fidelity, -np.array(grad_params)
        
    except Exception as e:
        # In case of solver failure (e.g. singularity)
        return 1.0, np.zeros(len(params))

if __name__ == "__main__":
    import yaml

    with open("configs/ms_2p_weak_blockade.yaml", 'r') as f:
        config = yaml.safe_load(f)
    h_params = config['hamiltonian_params']
    prop_params = PropagatorParameters(
        Nsteps=1024, Tstep=20.0/1024, Tcontrol=20.0,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )
    target_u = get_cz_target()
    bounds = [
        (10.0, 1000.0), # O_max
        (-5.0, 5.0),    # D_max
        (-5.0, 5.0),    # D_min
        (0.1, 5.0),     # tw
        (0.1, 20.0),    # tc
    ]
    best_infid = 1.0
    best_params = None
    for i in range(15):
        print(f"Attempt {i+1}/15...")
        guess = [np.random.uniform(b[0], b[1]) for b in bounds]
        res = scipy.optimize.minimize(
            objective_function, guess, args=(target_u, prop_params),
            jac=True, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_infid:
            best_infid = res.fun
            best_params = res.x
            print(f" New Best Infidelity: {best_infid:.6f}")
    print(f"\nGlobal Best Infidelity: {best_infid:.6f}")
    print(f"Optimal Parameters: {best_params}")
