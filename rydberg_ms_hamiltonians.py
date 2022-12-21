import numpy as np

def hamiltonian_ms_gate(h_params, Omega_eff=None):
    """
    Hamiltonian for the Mølmer-Sørensen gate using adiabatic Rydberg dressing.
    Basis: {|0,0>, |0,1>, |0,r>, |1,0>, |1,1>, |1,r>, |r,0>, |r,1>, |r,r>}
    """
    regime = h_params.get('regime', '1-photon')
    V = h_params.get('V_rr', 0.0)
    
    if regime == '1-photon':
        Delta = h_params.get('Delta_max', 0.0)
        if Omega_eff is None:
            Omega = h_params.get('Omega_max', 0.0)
        else:
            Omega = Omega_eff
    elif regime == '2-photon':
        # For 2-photon, the effective coupling is usually the result of the 2-photon process
        # Omega_eff = (Omega_1 * Omega_2) / (2 * Delta)
        Omega_1a = h_params.get('Omega_1a_max', 0.0)
        Omega_ar = h_params.get('Omega_ar', 0.0)
        Delta_1a = h_params.get('Delta_1a', 0.0)
        
        if Omega_eff is None:
            # Effective Rabi frequency for 2-photon transition
            Omega = (Omega_1a * Omega_ar) / (2 * Delta_1a) if Delta_1a != 0 else 0.0
        else:
            Omega = Omega_eff
        Delta = h_params.get('Delta_max', 0.0) # Fallback or specifically defined
    else:
        raise ValueError(f"Unsupported regime: {regime}")

    H = np.zeros((9, 9), dtype=complex)
    
    # Diagonal elements (energies)
    # Index mapping:
    # 0: 0,0 | 1: 0,1 | 2: 0,r | 3: 1,0 | 4: 1,1 | 5: 1,r | 6: r,0 | 7: r,1 | 8: r,r
    H[2, 2] = -Delta
    H[5, 5] = -Delta
    H[6, 6] = -Delta
    H[7, 7] = -Delta
    H[8, 8] = -2 * Delta + V
    
    # Off-diagonal elements (couplings)
    H[1, 2] = H[2, 1] = Omega / 2.0
    H[3, 6] = H[6, 3] = Omega / 2.0
    H[4, 5] = H[5, 4] = Omega / 2.0
    H[4, 7] = H[7, 4] = Omega / 2.0
    
    return H


def hamiltonian_ms_gate_grad(h_params, Omega_eff=None):
    """
    Gradient of the Hamiltonian with respect to Omega_eff.
    """
    H_grad = np.zeros((9, 9), dtype=complex)
    # The only terms that depend on Omega are the couplings
    H_grad[1, 2] = H_grad[2, 1] = 0.5
    H_grad[3, 6] = H_grad[6, 3] = 0.5
    H_grad[4, 5] = H_grad[5, 4] = 0.5
    H_grad[4, 7] = H_grad[7, 4] = 0.5
    return H_grad
