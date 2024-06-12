import numpy as np


def dagger(u):
    return np.transpose(np.conjugate(u))


def unitary_overlap(u, target, nstates=None):
    if nstates is None:
        nstates = min(np.linalg.matrix_rank(u), np.linalg.matrix_rank(target))
    return np.trace(np.dot(dagger(u), target)) / nstates


def unitary_fidelity(u, target, nstates=None):
    overlap = unitary_overlap(u, target, nstates)
    return np.abs(overlap)**2


def unitary_infidelity(u, target, nstates=None):
    return 1 - unitary_fidelity(u, target, nstates)


def state_fidelity(u, ket_initial, ket_target):
    bra_target = dagger(ket_target)
    return np.abs(np.dot(bra_target, np.dot(u, ket_initial))[0, 0])**2


def state_infidelity(u, ket_initial, ket_target):
    return 1 - state_fidelity(u, ket_initial, ket_target)


def state_infidelity_gradient(u, u_gradient, ket_initial, ket_target):
    bra_target = dagger(ket_target)
    overlap = np.dot(bra_target, np.dot(u, ket_initial))[0, 0]

    fidelity_gradient = np.empty(len(u_gradient))
    for n, du in enumerate(u_gradient):
        fidelity_gradient[n] = 2*np.real(
            np.conjugate(overlap) * np.dot(bra_target, np.dot(du, ket_initial))[0, 0])

    return -fidelity_gradient


def unitary_infidelity_gradient(u, u_gradient, target, nstates=None):
    if nstates is None:
        nstates = min(np.linalg.matrix_rank(u), np.linalg.matrix_rank(target))

    target_dagger = dagger(target)
    u_dagger = dagger(u)

    fidelity_gradient = np.empty(len(u_gradient))
    for n, du in enumerate(u_gradient):
        fidelity_gradient[n] = 2*np.real(
            np.trace(np.dot(target_dagger, du))
            * np.trace(np.dot(target, u_dagger))) / nstates**2

    return -fidelity_gradient
