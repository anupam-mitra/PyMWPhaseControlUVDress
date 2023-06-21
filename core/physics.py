import numpy as np


def hamiltonian_ms_gate(phi, h_params):
    """Hamiltonian for the MS gate in the 9D dressed basis."""
    regime = h_params.get('regime', '1-photon')
    V = h_params.get('V_rr', 0.0)

    if isinstance(phi, (list, tuple, np.ndarray)) and len(phi) == 2:
        omega, delta = phi
    else:
        omega = phi
        delta = h_params.get('Delta_max', 0.0)

    if regime == '2-photon':
        Omega_ar = h_params.get('Omega_ar', 0.0)
        Delta_1a = h_params.get('Delta_1a', 0.0)
        omega = (omega * Omega_ar) / (2 * Delta_1a) if Delta_1a != 0 else 0.0

    H = np.zeros((9, 9), dtype=complex)
    H[2, 2] = H[5, 5] = H[6, 6] = H[7, 7] = -delta
    H[8, 8] = -2 * delta + V
    H[1, 2] = H[2, 1] = omega / 2.0
    H[3, 6] = H[6, 3] = omega / 2.0
    H[4, 5] = H[5, 4] = omega / 2.0
    H[4, 7] = H[7, 4] = omega / 2.0
    return H


def hamiltonian_ms_gate_grad(phi, h_params):
    """Gradient of the MS gate Hamiltonian."""
    regime = h_params.get('regime', '1-photon')
    is_pair = isinstance(phi, (list, tuple, np.ndarray)) and len(np.atleast_1d(phi)) == 2

    factor_omega = 1.0
    if regime == '2-photon':
        Omega_ar = h_params.get('Omega_ar', 0.0)
        Delta_1a = h_params.get('Delta_1a', 0.0)
        factor_omega = Omega_ar / (2 * Delta_1a) if Delta_1a != 0 else 0.0

    H_grad_omega = np.zeros((9, 9), dtype=complex)
    H_grad_omega[1, 2] = H_grad_omega[2, 1] = 0.5 * factor_omega
    H_grad_omega[3, 6] = H_grad_omega[6, 3] = 0.5 * factor_omega
    H_grad_omega[4, 5] = H_grad_omega[5, 4] = 0.5 * factor_omega
    H_grad_omega[4, 7] = H_grad_omega[7, 4] = 0.5 * factor_omega

    if not is_pair:
        return H_grad_omega

    H_grad_delta = np.zeros((9, 9), dtype=complex)
    H_grad_delta[2, 2] = H_grad_delta[5, 5] = H_grad_delta[6, 6] = H_grad_delta[7, 7] = -1.0
    H_grad_delta[8, 8] = -2.0
    return [H_grad_omega, H_grad_delta]
