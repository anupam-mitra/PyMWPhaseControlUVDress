import itertools

import numpy as np
import scipy.linalg
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def dagger(u):
    return np.transpose(np.conjugate(u))


def propagator_step_derivative(h_step, h_step_deriv, Tstep):
    eigenvalues, eigenvectors = scipy.linalg.eigh(h_step)
    u_step_derivative = np.zeros(eigenvectors.shape, dtype=complex)

    for (a, ea), (b, eb) in itertools.product(enumerate(eigenvalues), enumerate(eigenvalues)):
        outerprod = np.outer(eigenvectors[:, a], eigenvectors[:, b].conjugate())
        h_step_grad_matrixel = np.dot(eigenvectors[:, a].conjugate(), np.dot(h_step_deriv, eigenvectors[:, b]))
        if ea == eb:
            u_step_derivative += -1j * np.exp(-1j * Tstep * ea) * outerprod * h_step_grad_matrixel
        else:
            u_step_derivative += (
                (np.exp(-1j * Tstep * ea) - np.exp(-1j * Tstep * eb)) / (ea - eb)
            ) * outerprod * h_step_grad_matrixel

    return u_step_derivative


def _propagator_from_steps(u_steps):
    dimensions = u_steps[0].shape[0]
    u = np.identity(dimensions, dtype=complex)
    for u_step in u_steps:
        u = np.dot(u_step, u)
    return u


def _gradient_from_steps(h_steps, u_steps, h_grad_steps, Tstep):
    dimensions = h_steps[0].shape[0]
    u_gradient = []

    for n in range(len(h_steps)):
        u_gradient.append(np.identity(dimensions, dtype=complex))
        for m in range(n):
            u_gradient[n] = np.dot(u_steps[m], u_gradient[n])

        u_step_derivative = propagator_step_derivative(h_steps[n], h_grad_steps[n], Tstep)
        u_gradient[n] = np.dot(u_step_derivative, u_gradient[n])

        for m in range(n + 1, len(h_steps)):
            u_gradient[n] = np.dot(u_steps[m], u_gradient[n])

    return u_gradient


def _propagator_steps(phi, propagator_params, gradient=False):
    if isinstance(propagator_params, dict):
        Nsteps = propagator_params['Nsteps']
        Tstep = propagator_params['Tstep']
        hamiltonian_func = propagator_params['HamiltonianMatrix']
        h_params = propagator_params['HamiltonianParameters']
        hamiltonian_grad_func = propagator_params.get('HamiltonianMatrixGradient')
    else:
        Nsteps = propagator_params.Nsteps
        Tstep = propagator_params.Tstep
        hamiltonian_func = propagator_params.hamiltonian_matrix_func
        h_params = propagator_params.hamiltonian_params
        hamiltonian_grad_func = propagator_params.hamiltonian_matrix_grad_func

    h_steps = [hamiltonian_func(phi[n], h_params) for n in range(Nsteps)]
    u_steps = [expm(-1j * Tstep * h) for h in h_steps]

    if not gradient:
        return h_steps, u_steps

    h_grad_steps = [hamiltonian_grad_func(phi[n], h_params) for n in range(Nsteps)]
    return h_steps, u_steps, h_grad_steps


def propagator(phi, propagator_params):
    h_steps, u_steps = _propagator_steps(phi, propagator_params)
    return _propagator_from_steps(u_steps)


def propagator_gradient(phi, propagator_params):
    if isinstance(propagator_params, dict):
        Tstep = propagator_params['Tstep']
    else:
        Tstep = propagator_params.Tstep
    h_steps, u_steps, h_grad_steps = _propagator_steps(phi, propagator_params, True)
    return _gradient_from_steps(h_steps, u_steps, h_grad_steps, Tstep)


def propagator_with_gradient(phi, propagator_params):
    if isinstance(propagator_params, dict):
        Tstep = propagator_params['Tstep']
    else:
        Tstep = propagator_params.Tstep
    h_steps, u_steps, h_grad_steps = _propagator_steps(phi, propagator_params, True)
    return _propagator_from_steps(u_steps), _gradient_from_steps(h_steps, u_steps, h_grad_steps, Tstep)


def _get_hamiltonian_at_t(t, phi, propagator_params):
    if callable(phi):
        phi_val = phi(t, propagator_params.hamiltonian_params)
    elif isinstance(phi, (list, np.ndarray)):
        Nsteps = propagator_params.Nsteps
        Tcontrol = propagator_params.Tcontrol
        n = int(np.floor(t / (Tcontrol / Nsteps)))
        if n >= Nsteps:
            n = Nsteps - 1
        if n < 0:
            n = 0
        phi_val = phi[n]
    else:
        raise TypeError(f'phi must be a callable or a la-sequence, got {type(phi)}')

    return propagator_params.hamiltonian_matrix_func(phi_val, propagator_params.hamiltonian_params)


def _ivp_system(t, y, phi, propagator_params, compute_gradient=False, params_vec=None):
    if callable(phi):
        phi_val = phi(t, propagator_params.hamiltonian_params)
    elif isinstance(phi, (list, np.ndarray)):
        Tcontrol = propagator_params.Tcontrol
        Nsteps = propagator_params.Nsteps
        n = int(np.floor(t / (Tcontrol / Nsteps)))
        if n >= Nsteps:
            n = Nsteps - 1
        if n < 0:
            n = 0
        phi_val = phi[n]
    else:
        raise TypeError(f'phi must be a callable or a la-sequence, got {type(phi)}')

    h_t = propagator_params.hamiltonian_matrix_func(phi_val, propagator_params.hamiltonian_params)
    ndim = h_t.shape[0]

    U_slice = y[0:ndim * ndim]
    U = U_slice.reshape((ndim, ndim))
    dUdt = -1j * np.dot(h_t, U)

    if compute_gradient and params_vec is not None:
        h_grad_func = propagator_params.hamiltonian_matrix_grad_func
        h_grads = h_grad_func(t, phi_val, propagator_params.hamiltonian_params, params_vec)
        dVdts = []
        for k in range(len(params_vec)):
            V_k = y[ndim * ndim + k * ndim * ndim: ndim * ndim + (k + 1) * ndim * ndim].reshape((ndim, ndim))
            dVkdt = -1j * (np.dot(h_grads[k], U) + np.dot(h_t, V_k))
            dVdts.append(dVkdt.flatten())
        return np.concatenate([dUdt.flatten(), *dVdts])

    return dUdt.flatten()


def propagator_ivp(phi, propagator_params):
    Tcontrol = propagator_params.Tcontrol
    ndim = _get_hamiltonian_at_t(0, phi, propagator_params).shape[0]
    y0 = np.eye(ndim, dtype=complex).flatten()
    sol = solve_ivp(
        _ivp_system,
        (0, Tcontrol),
        y0,
        args=(phi, propagator_params, False, None),
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
    )
    return sol.y[:, -1].reshape((ndim, ndim))


def propagator_with_gradient_ivp(phi, propagator_params, params_vec):
    Tcontrol = propagator_params.Tcontrol
    ndim = _get_hamiltonian_at_t(0, phi, propagator_params).shape[0]
    Nparams = len(params_vec)
    y0 = np.concatenate([
        np.eye(ndim, dtype=complex).flatten(),
        np.zeros(Nparams * ndim * ndim, dtype=complex),
    ])
    sol = solve_ivp(
        _ivp_system,
        (0, Tcontrol),
        y0,
        args=(phi, propagator_params, True, params_vec),
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
    )
    final_state = sol.y[:, -1]
    U_final = final_state[0:ndim * ndim].reshape((ndim, ndim))
    V_grads = []
    for k in range(Nparams):
        V_k = final_state[ndim * ndim + k * ndim * ndim: ndim * ndim + (k + 1) * ndim * ndim].reshape((ndim, ndim))
        V_grads.append(V_k)
    return U_final, V_grads


def calc_time_evolve_op(hamiltonians, grad_hamiltonians, tsteps, gradient=True):
    hamiltonians_shape_tuple = np.shape(hamiltonians)
    grad_hamiltonians_shape_tuple = np.shape(grad_hamiltonians)

    assert hamiltonians_shape_tuple[0] == hamiltonians_shape_tuple[1] == grad_hamiltonians_shape_tuple[0] == grad_hamiltonians_shape_tuple[1]
    assert hamiltonians_shape_tuple[2] == grad_hamiltonians_shape_tuple[2]
    assert hamiltonians_shape_tuple[3] == grad_hamiltonians_shape_tuple[3]

    ndim = hamiltonians_shape_tuple[0]
    nlandmarks = hamiltonians_shape_tuple[2]
    nsteps = hamiltonians_shape_tuple[3]
    ncontrols = grad_hamiltonians_shape_tuple[4]

    propagators = np.zeros((ndim, ndim, nlandmarks, nsteps + 1, nsteps + 1), dtype=complex)
    grad_propagators = np.zeros((ndim, ndim, nlandmarks, nsteps + 1, nsteps + 1, nsteps, ncontrols), dtype=complex)

    for l in range(nlandmarks):
        propagators[:, :, l, 0, 0] = np.eye(ndim, ndim)

        for s_begin in range(nsteps):
            propagators[:, :, l, s_begin + 1, s_begin + 1] = np.eye(ndim, ndim)
            propagators[:, :, l, s_begin + 1, s_begin] = expm(-1j * tsteps[s_begin] * hamiltonians[:, :, l, s_begin])
            propagators[:, :, l, s_begin, s_begin + 1] = dagger(propagators[:, :, l, s_begin + 1, s_begin])

        for s_begin in range(nsteps):
            for s_end in range(s_begin + 1, nsteps + 1):
                propagators[:, :, l, s_end, s_begin] = np.dot(propagators[:, :, l, s_end, s_end - 1], propagators[:, :, l, s_end - 1, s_begin])
                propagators[:, :, l, s_begin, s_end] = dagger(propagators[:, :, l, s_end, s_begin])

        for k in range(ncontrols):
            for s_begin in range(nsteps):
                grad_propagators[:, :, l, s_begin + 1, s_begin, s_begin, k] = propagator_step_derivative(
                    hamiltonians[:, :, l, s_begin],
                    grad_hamiltonians[:, :, l, s_begin, k],
                    tsteps[s_begin],
                )
                grad_propagators[:, :, l, s_begin, s_begin + 1, s_begin, k] = dagger(
                    grad_propagators[:, :, l, s_begin + 1, s_begin, s_begin, k]
                )

            for s_begin in range(nsteps):
                for s_end in range(s_begin + 1, nsteps + 1):
                    for s in range(s_begin, s_end):
                        grad_propagators[:, :, l, s_end, s_begin, s, k] = np.dot(
                            propagators[:, :, l, s_end, s + 1],
                            np.dot(grad_propagators[:, :, l, s + 1, s, s, k], propagators[:, :, l, s, s_begin]),
                        )
                        grad_propagators[:, :, l, s_begin, s_end, s, k] = dagger(grad_propagators[:, :, l, s_end, s_begin, s, k])

    return (propagators, grad_propagators) if gradient else propagators
