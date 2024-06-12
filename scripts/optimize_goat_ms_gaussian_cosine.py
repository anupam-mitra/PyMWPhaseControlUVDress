import numpy as np
import scipy.optimize
from core.evolution import propagator_with_gradient_ivp
from core.ms_goat_models import GaussianCosineAnsatz
from core.params import PropagatorParameters
from core.unitaries import get_ms_yy_target

def combined_ansatz_val(t, params, T_control, n_basis):
    # params: [A_1, B_1, ... A_N, B_N] for Omega, then [A_1, B_1, ... A_N, B_N] for Delta
    # Total params: 2 * 2 * n_basis
    half = 2 * n_basis
    params_omega = params[:half]
    params_delta = params[half:]
    
    # Fixed Gaussian window for both
    tc = T_control / 2.0
    w = T_control / 4.0
    
    ansatz_omega = GaussianCosineAnsatz(T_control, n_basis, params_omega, tc=tc, w=w)
    ansatz_delta = GaussianCosineAnsatz(T_control, n_basis, params_delta, tc=tc, w=w)
    
    return np.array([ansatz_omega.evaluate(t), ansatz_delta.evaluate(t)])

def combined_ansatz_grad(t, params, T_control, n_basis):
    half = 2 * n_basis
    params_omega = params[:half]
    params_delta = params[half:]
    
    tc = T_control / 2.0
    w = T_control / 4.0
    
    ansatz_omega = GaussianCosineAnsatz(T_control, n_basis, params_omega, tc=tc, w=w)
    ansatz_delta = GaussianCosineAnsatz(T_control, n_basis, params_delta, tc=tc, w=w)
    
    grad_omega = ansatz_omega.gradient(t) # (2*n_basis,)
    grad_delta = ansatz_delta.gradient(t) # (2*n_basis,)
    
    # We want (2, 4*n_basis)
    res = np.zeros((2, 4 * n_basis))
    res[0, :half] = grad_omega
    res[1, half:] = grad_delta
    return res

def goat_hamiltonian(phi_val, h_params):
    if isinstance(phi_val, np.ndarray):
        phi_val = phi_val.item() if phi_val.size == 1 else phi_val
        
    # phi_val is [omega, delta]
    omega, delta = phi_val
    V = h_params.get('V_rr', 0.0)
    
    H = np.zeros((9, 9), dtype=complex)
    H[2, 2] = H[5, 5] = H[6, 6] = H[7, 7] = -delta
    H[8, 8] = -2 * delta + V
    H[1, 2] = H[2, 1] = omega / 2.0
    H[3, 6] = H[6, 3] = omega / 2.0
    H[4, 5] = H[5, 4] = omega / 2.0
    H[4, 7] = H[7, 4] = omega / 2.0
    return H

def goat_hamiltonian_grad(t, phi_val, h_params, params_vec):
    # phi_val is [omega, delta]
    # params_vec is [A_1...B_N (omega), A_1...B_N (delta)]
    
    T_control = T_CONTROL
    n_basis = len(params_vec) // 4
    
    H_grad_omega = np.zeros((9, 9), dtype=complex)
    H_grad_omega[1, 2] = H_grad_omega[2, 1] = 0.5
    H_grad_omega[3, 6] = H_grad_omega[6, 3] = 0.5
    H_grad_omega[4, 5] = H_grad_omega[5, 4] = 0.5
    H_grad_omega[4, 7] = H_grad_omega[7, 4] = 0.5
    
    H_grad_delta = np.zeros((9, 9), dtype=complex)
    H_grad_delta[2, 2] = H_grad_delta[5, 5] = H_grad_delta[6, 6] = H_grad_delta[7, 7] = -1.0
    H_grad_delta[8, 8] = -2.0
    
    phi_grads = combined_ansatz_grad(t, params_vec, T_control, n_basis)
    
    res_grads = []
    for k in range(len(params_vec)):
        H_k = phi_grads[0, k] * H_grad_omega + phi_grads[1, k] * H_grad_delta
        res_grads.append(H_k)
        
    return res_grads

# Global for the ansatz call
CURRENT_PARAMS = None
T_CONTROL = 2.0
N_BASIS = 5

def phi_ansatz_wrapper(t, h_params):
    return combined_ansatz_val(t, CURRENT_PARAMS, T_CONTROL, N_BASIS)

from core.evolution import propagator_with_gradient

def objective_function(params, target_u, prop_params):
    global CURRENT_PARAMS
    CURRENT_PARAMS = params
    
    U, V_grads = propagator_with_gradient_ivp(phi_ansatz_wrapper, prop_params, params)
    
    overlap = np.trace(np.dot(target_u.conj().T, U))
    fidelity = (1.0/81.0) * np.abs(overlap)**2
    
    grad_params = []
    for V_k in V_grads:
        term = np.trace(np.dot(target_u.conj().T, V_k))
        grad_params.append((2.0/81.0) * (overlap.conj() * term).real)
        
    return 1.0 - fidelity, -np.array(grad_params)



def run_optimization(T_val, n_basis):
    global T_CONTROL, N_BASIS, CURRENT_PARAMS
    T_CONTROL = T_val
    N_BASIS = n_basis
    
    h_params = {'V_rr': 1.0}
    prop_params = PropagatorParameters(
        Nsteps=50,
        Tstep=T_val/50,
        Tcontrol=T_val,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )
    
    target_u = get_target_u()
    np.random.seed(42)
    initial_guess = np.random.uniform(-1, 1, 4 * n_basis)
    
    res = scipy.optimize.minimize(
        objective_function, 
        initial_guess, 
        args=(target_u, prop_params),
        jac=True, 
        method='L-BFGS-B'
    )
    return res.fun

if __name__ == "__main__":
    import scipy.optimize
    
    V_rr = 1.0
    T_CONTROL_VAL = 81 * 2 * np.pi / V_rr  # Large T_control
    N_BASIS_VAL = 81       # dim^2
    
    # Update globals used by phi_ansatz_wrapper
    T_CONTROL = T_CONTROL_VAL
    N_BASIS = N_BASIS_VAL
    
    h_params = {'V_rr': V_rr}
    # For IVP, Nsteps doesn't dictate the grid but we provide it for compatibility
    prop_params = PropagatorParameters(
        Nsteps=1000, # Increase Nsteps to ensure solve_ivp has enough context if needed
        Tstep=T_CONTROL_VAL/1000,
        Tcontrol=T_CONTROL_VAL,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=goat_hamiltonian,
        hamiltonian_matrix_grad_func=goat_hamiltonian_grad
    )

    
    target_u = get_target_u()
    
    # Initial guess: small random values for A_n, B_n for both Omega and Delta
    # Total params = 2 * (2 * N_BASIS) = 4 * N_BASIS
    np.random.seed(42)
    initial_guess = np.random.uniform(-0.5, 0.5, 4 * N_BASIS_VAL)
    
    print(f"Starting GOAT optimization with N_BASIS={N_BASIS_VAL} and T_control={T_CONTROL_VAL}...")
    res = scipy.optimize.minimize(
        objective_function, 
        initial_guess, 
        args=(target_u, prop_params),
        jac=True, 
        method='L-BFGS-B'
    )
    print("Optimization Result:\n", res)
