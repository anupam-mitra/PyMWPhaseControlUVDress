import numpy as np
import scipy.optimize
import copy
from core.evolution import propagator_with_gradient_ivp
from params import PropagatorParameters
import core.physics as rydberg_ms_hamiltonians

def get_ms_yy_target():
    """
    Returns the MS_yy target unitary in the 9D basis.
    Basis: {|0,0>, |0,1>, |0,r>, |1,0>, |1,1>, |1,r>, |r,0>, |r,1>, |r,r>}
    Qubit subspace indices: [0, 1, 3, 4]
    """
    U_target = np.eye(9, dtype=complex)
    q = [0, 1, 3, 4]
    sy_sy = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0]
    ])
    coeff = 1.0 / np.sqrt(2)
    for i in range(4):
        for j in range(4):
            val = 0
            if i == j:
                val += coeff * 1.0
            val -= coeff * 1j * sy_sy[i, j]
            U_target[q[i], q[j]] = val
    return U_target

def two_photon_omega_ramp(t, omega_max, t_stop, tw):
    t_abs = np.abs(t)
    if t_abs <= t_stop:
        return omega_max
    else:
        return omega_max * np.exp(-(t_abs - t_stop)**2 / (2 * tw**2))

def two_photon_omega_grad(t, omega_max, t_stop, tw):
    t_abs = np.abs(t)
    grads = np.zeros(3)
    if t_abs <= t_stop:
        grads[0] = 1.0
        grads[1] = 0.0
        grads[2] = 0.0
    else:
        val = np.exp(-(t_abs - t_stop)**2 / (2 * tw**2))
        grads[0] = val
        grads[1] = omega_max * val * (t_abs - t_stop) / (tw**2) * np.sign(t)
        grads[2] = omega_max * val * (t_abs - t_stop)**2 / (tw**3)
    return grads

def objective_function(params, target_u, base_prop_params):
    """
    Cost function for MS gate using IVP evolution.
    params: [omega_max, t_stop, tw, delta_1a]
    """
    omega_max, t_stop, tw, delta_1a = params
    
    def compute_fidelity(current_params):
        om_max, ts, tw_val, d1a = current_params
        prop_params = copy.deepcopy(base_prop_params)
        prop_params.hamiltonian_params['Omega_1a_max'] = om_max
        prop_params.hamiltonian_params['Delta_1a'] = d1a
        
        Nsteps = prop_params.Nsteps
        Tcontrol = prop_params.Tcontrol
        t_grid = np.linspace(-Tcontrol/2, Tcontrol/2, Nsteps)
        phi = np.array([two_photon_omega_ramp(t, om_max, ts, tw_val) for t in t_grid])
        
        U, _ = propagator_with_gradient_ivp(phi, prop_params)
        overlap = np.trace(np.dot(target_u.conj().T, U))
        return (1.0/81.0) * np.abs(overlap)**2

    # Main call for value and la-gradients
    prop_params = copy.deepcopy(base_prop_params)
    prop_params.hamiltonian_params['Omega_1a_max'] = omega_max
    prop_params.hamiltonian_params['Delta_1a'] = delta_1a
    Nsteps = prop_params.Nsteps
    Tcontrol = prop_params.Tcontrol
    t_grid = np.linspace(-Tcontrol/2, Tcontrol/2, Nsteps)
    phi = np.array([two_photon_omega_ramp(t, omega_max, t_stop, tw) for t in t_grid])
    
    U, V_grads = propagator_with_gradient_ivp(phi, prop_params)
    overlap = np.trace(np.dot(target_u.conj().T, U))
    fidelity = (1.0/81.0) * np.abs(overlap)**2
    
    dF_dphi = []
    for V_n in V_grads:
        term = np.trace(np.dot(target_u.conj().T, V_n))
        dF_dphi.append((2.0/81.0) * (overlap.conj() * term).real)
    
    grad_params = np.zeros(4)
    for n in range(Nsteps):
        t = t_grid[n]
        phi_grad = two_photon_omega_grad(t, omega_max, t_stop, tw)
        grad_params[0:3] += dF_dphi[n] * phi_grad
        
    # Finite difference for Delta_1a
    eps = 1e-4
    f_plus = compute_fidelity([omega_max, t_stop, tw, delta_1a + eps])
    f_minus = compute_fidelity([omega_max, t_stop, tw, delta_1a - eps])
    grad_params[3] = (f_plus - f_minus) / (2 * eps)
    
    return 1.0 - fidelity, -grad_params


if __name__ == "__main__":
    h_params = {
        'regime': '2-photon',
        'V_rr': 1.0,
        'Omega_ar': 1.0,
        'Delta_1a': -1.0,
        'Delta_max': 0.1,
    }
    prop_params = PropagatorParameters(
        Nsteps=50,
        Tstep=0.04,
        Tcontrol=2.0,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=rydberg_ms_hamiltonians.hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=rydberg_ms_hamiltonians.hamiltonian_ms_gate_grad
    )
    target_u = get_ms_yy_target()
    initial_guess = [1.5, 0.4, 0.3, -1.0]
    bounds = [(0.1, 10.0), (0.01, 2.0), (0.01, 2.0), (-10.0, 10.0)]
    
    print("Starting optimization...")
    res = scipy.optimize.minimize(
        objective_function, 
        initial_guess, 
        args=(target_u, prop_params),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'gtol': 1e-7}
    )
    print("Optimization Result:\n", res)
    print("Optimal Parameters:", res.x)
    print("Final Infidelity:", res.fun)
