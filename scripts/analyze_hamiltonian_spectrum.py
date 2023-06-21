import numpy as np

from core.physics import hamiltonian_ms_gate as goat_hamiltonian


def get_eigenvalues(omega, delta, V_rr=1.0):
    h_params = {'V_rr': V_rr}
    H = goat_hamiltonian([omega, delta], h_params)
    return np.linalg.eigvalsh(H)


if __name__ == '__main__':
    test_cases = [
        (1.0, -0.1),
        (1.0, -1.0),
        (5.0, -0.1),
        (5.0, -5.0),
    ]

    print(f"{'Omega':<10} | {'Delta':<10} | {'Eigenvalues'}")
    print('-' * 50)
    for omega, delta in test_cases:
        evals = get_eigenvalues(omega, delta)
        evals_str = ', '.join([f'{e:.3f}' for e in evals])
        print(f"{omega:<10.2f} | {delta:<10.2f} | {evals_str}")
