import numpy as np

from core.evolution import propagator_ivp
from core.params import PropagatorParameters
from core.physics import hamiltonian_ms_gate


def get_ms_yy_target():
    U_target = np.eye(9, dtype=complex)
    q = [0, 1, 3, 4]
    sy_sy = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
    ])
    coeff = 1.0 / np.sqrt(2)
    for i in range(4):
        for j in range(4):
            val = coeff if i == j else 0
            val -= coeff * 1j * sy_sy[i, j]
            U_target[q[i], q[j]] = val
    return U_target


def get_ms_yy_target_4x4():
    target = np.array([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
    ], dtype=complex)
    return np.cos(np.pi / 4.0) * np.eye(4, dtype=complex) - 1j * np.sin(np.pi / 4.0) * target


def get_cz_target():
    U_target = np.eye(9, dtype=complex)
    U_target[4, 4] = -1.0
    return U_target


def unitary_infidelity(U_final, target_u):
    overlap = np.trace(np.dot(target_u.conj().T, U_final))
    fidelity = (1.0 / 81.0) * np.abs(overlap) ** 2
    return 1.0 - np.clip(fidelity, 0.0, 1.0)


def unitary_infidelity_4x4(U_final_4x4, target_u_4x4=None):
    if target_u_4x4 is None:
        target_u_4x4 = get_ms_yy_target_4x4()
    overlap = np.trace(np.dot(target_u_4x4.conj().T, U_final_4x4))
    fidelity = (1.0 / 16.0) * np.abs(overlap) ** 2
    return 1.0 - np.clip(fidelity, 0.0, 1.0)


def _single_qubit_xy_pulse(theta, phase=0.0):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    e_plus = np.exp(1j * phase)
    e_minus = np.exp(-1j * phase)
    u2 = np.array([
        [c, -1j * e_minus * s],
        [-1j * e_plus * s, c],
    ], dtype=complex)
    return np.array([
        [u2[0, 0], u2[0, 1], 0],
        [u2[1, 0], u2[1, 1], 0],
        [0, 0, 1],
    ], dtype=complex)


def microwave_x_pulse_9d(theta, phase=0.0):
    u1 = _single_qubit_xy_pulse(theta, phase)
    return np.kron(u1, u1)


def spin_echo_unitary_from_ramp(U_ramp, phase=0.0):
    sequence = [
        microwave_x_pulse_9d(np.pi / 2.0, phase),
        U_ramp,
        microwave_x_pulse_9d(np.pi, phase),
        U_ramp,
        microwave_x_pulse_9d(np.pi / 2.0, phase),
    ]
    u = np.eye(9, dtype=complex)
    for step in sequence:
        u = np.dot(step, u)
    return u


def one_photon_ramp(t, params):
    t_const = params['t_constant_duration']
    t_w = params['t_gaussian_width']
    t_mid = params.get('t_mid', 0.0)
    n_gaussian_widths = params.get('n_gaussian_widths', 2)

    t_gaussian_duration = n_gaussian_widths * t_w
    t_dress_begin = t_mid - t_gaussian_duration - t_const / 2.0
    t_dress_end = t_mid - t_const / 2.0
    t_undress_begin = t_mid + t_const / 2.0
    t_undress_end = t_mid + t_gaussian_duration + t_const / 2.0

    delta_max = params['Delta_max']
    delta_min = params['Delta_min']
    omega_max = params['Omega_max']
    omega_min = params['Omega_min']

    if t < t_dress_begin or t > t_undress_end:
        return omega_min, delta_max
    if t < t_dress_end:
        t_zeroed = t - t_dress_end
        gaussian_factor = np.exp(-t_zeroed ** 2 / (2.0 * t_w ** 2))
        omega = omega_min + (omega_max - omega_min) * gaussian_factor
        delta = delta_max + (delta_min - delta_max) / (t_dress_end - t_dress_begin) * (t - t_dress_begin)
        return omega, delta
    if t > t_undress_begin:
        t_zeroed = t - t_undress_begin
        gaussian_factor = np.exp(-t_zeroed ** 2 / (2.0 * t_w ** 2))
        omega = omega_min + (omega_max - omega_min) * gaussian_factor
        delta = delta_min + (delta_max - delta_min) / (t_undress_end - t_undress_begin) * t_zeroed
        return omega, delta
    return omega_max, delta_min


def two_photon_omega_ramp(t, omega_max, t_stop, tw):
    t_abs = np.abs(t)
    if t_abs <= t_stop:
        return omega_max
    return omega_max * np.exp(-(t_abs - t_stop) ** 2 / (2.0 * tw ** 2))


def _two_photon_t_stop(pulse_params):
    return pulse_params.get('t_stop', pulse_params.get('t_constant_duration', 0.0) / 2.0)


def evaluate_one_photon_ms_infidelity(pulse_params, h_params, target_u=None, nsteps=200):
    if target_u is None:
        target_u = get_ms_yy_target()
    Tcontrol = pulse_params['Tcontrol']
    prop_params = PropagatorParameters(
        Nsteps=nsteps,
        Tstep=Tcontrol / nsteps,
        Tcontrol=Tcontrol,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )
    t_grid = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, nsteps)
    phi = [one_photon_ramp(t, pulse_params) for t in t_grid]
    U_final = propagator_ivp(phi, prop_params)
    return unitary_infidelity(U_final, target_u)


def evaluate_one_photon_ms_spin_echo_infidelity(pulse_params, h_params, target_u_4x4=None, nsteps=200):
    if target_u_4x4 is None:
        target_u_4x4 = get_ms_yy_target_4x4()
    Tcontrol = pulse_params['Tcontrol']
    prop_params = PropagatorParameters(
        Nsteps=nsteps,
        Tstep=Tcontrol / nsteps,
        Tcontrol=Tcontrol,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )
    t_grid = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, nsteps)
    phi = [one_photon_ramp(t, pulse_params) for t in t_grid]
    U_ramp = propagator_ivp(phi, prop_params)
    U_echo = spin_echo_unitary_from_ramp(U_ramp)
    q = [0, 1, 3, 4]
    Uq = U_echo[np.ix_(q, q)]
    return unitary_infidelity_4x4(Uq, target_u_4x4), U_echo


def evaluate_two_photon_ms_infidelity(pulse_params, h_params, target_u=None, nsteps=200):
    if target_u is None:
        target_u = get_ms_yy_target()
    Tcontrol = pulse_params['Tcontrol']
    eff_h_params = dict(h_params)
    eff_h_params['regime'] = '1-photon'
    prop_params = PropagatorParameters(
        Nsteps=nsteps,
        Tstep=Tcontrol / nsteps,
        Tcontrol=Tcontrol,
        hamiltonian_params=eff_h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )
    t_grid = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, nsteps)
    t_stop = _two_photon_t_stop(pulse_params)
    phi = np.array([two_photon_omega_ramp(t, pulse_params['Omega_1a_max'], t_stop, pulse_params['t_gaussian_width']) for t in t_grid])
    U_final = propagator_ivp(phi, prop_params)
    return unitary_infidelity(U_final, target_u)


def two_photon_effective_ramp(t, pulse_params, h_params):
    omega_1a = two_photon_omega_ramp(t, pulse_params['Omega_1a_max'], _two_photon_t_stop(pulse_params), pulse_params['t_gaussian_width'])
    omega_ar = h_params['Omega_ar']
    delta_1a = h_params['Delta_1a']
    delta_ar = h_params['Delta_ar']
    omega_eff = (omega_1a * omega_ar) / (2.0 * delta_1a)
    delta_eff = delta_1a + delta_ar + (omega_1a ** 2) / (4.0 * delta_1a) - (omega_ar ** 2) / (4.0 * delta_ar)
    return np.array([omega_eff, delta_eff])


def evaluate_two_photon_ms_spin_echo_infidelity(pulse_params, h_params, target_u_4x4=None, nsteps=200):
    if target_u_4x4 is None:
        target_u_4x4 = get_ms_yy_target_4x4()
    Tcontrol = pulse_params['Tcontrol']
    eff_h_params = dict(h_params)
    eff_h_params['regime'] = '1-photon'
    prop_params = PropagatorParameters(
        Nsteps=nsteps,
        Tstep=Tcontrol / nsteps,
        Tcontrol=Tcontrol,
        hamiltonian_params=eff_h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )
    t_grid = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, nsteps)
    phi = np.array([two_photon_effective_ramp(t, pulse_params, h_params) for t in t_grid])
    U_ramp = propagator_ivp(phi, prop_params)
    U_echo = spin_echo_unitary_from_ramp(U_ramp)
    q = [0, 1, 3, 4]
    Uq = U_echo[np.ix_(q, q)]
    return unitary_infidelity_4x4(Uq, target_u_4x4), U_echo
