import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from core.unitaries import get_ms_yy_target_4x4, microwave_x_pulse_9d
from core.params import PropagatorParameters
from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad
from core.evolution import propagator_ivp, propagator_with_gradient_ivp


V_RR = 2.0 * np.pi
TIME_UNIT = 2.0 * np.pi / V_RR
THRESHOLD = 1e-4
QUBIT_IDXS = [0, 1, 3, 4]
M_PI_2 = microwave_x_pulse_9d(np.pi / 2.0)
M_PI = microwave_x_pulse_9d(np.pi)
TARGET_4X4 = get_ms_yy_target_4x4()


def _spin_echo_metrics(U_ramp, V_grads):
    U_echo = M_PI_2 @ U_ramp @ M_PI @ U_ramp @ M_PI_2
    Uq = U_echo[np.ix_(QUBIT_IDXS, QUBIT_IDXS)]
    overlap = np.trace(TARGET_4X4.conj().T @ Uq)
    fidelity = (1.0 / 16.0) * np.abs(overlap) ** 2
    infidelity = 1.0 - np.clip(fidelity, 0.0, 1.0)

    grad = np.zeros(len(V_grads))
    for k, V_k in enumerate(V_grads):
        dU = M_PI_2 @ V_k @ M_PI @ U_ramp @ M_PI_2 + M_PI_2 @ U_ramp @ M_PI @ V_k @ M_PI_2
        dUq = dU[np.ix_(QUBIT_IDXS, QUBIT_IDXS)]
        term = np.trace(TARGET_4X4.conj().T @ dUq)
        grad[k] = -(2.0 / 16.0) * (overlap.conj() * term).real

    return float(infidelity), grad, U_echo


def _one_photon_controls_and_grad_fixed_omega(t, params, omega_max, n_gaussian_widths, t_mid=0.0):
    delta_max, delta_min, t_gaussian_width, t_constant_duration = params
    t_gaussian_duration = n_gaussian_widths * t_gaussian_width

    t_dress_begin = t_mid - t_gaussian_duration - t_constant_duration / 2.0
    t_dress_end = t_mid - t_constant_duration / 2.0
    t_undress_begin = t_mid + t_constant_duration / 2.0
    t_undress_end = t_mid + t_gaussian_duration + t_constant_duration / 2.0

    grad = np.zeros((2, 4))

    if t < t_dress_begin or t > t_undress_end:
        grad[1, 0] = 1.0
        return np.array([0.0, delta_max]), grad

    if t < t_dress_end:
        gauss = np.exp(-(t - t_dress_end) ** 2 / (2.0 * t_gaussian_width ** 2))
        omega = omega_max * gauss
        grad[1, 0] = 1.0 - (t - t_dress_begin) / t_gaussian_duration
        grad[1, 1] = (t - t_dress_begin) / t_gaussian_duration
        grad[1, 2] = -(delta_min - delta_max) * (t - t_mid + t_constant_duration / 2.0) / (n_gaussian_widths * t_gaussian_width ** 2)
        grad[1, 3] = (delta_min - delta_max) / (2.0 * t_gaussian_duration)
        return np.array([omega, delta_max + (delta_min - delta_max) * (t - t_dress_begin) / t_gaussian_duration]), grad

    if t <= t_undress_begin:
        grad[1, 1] = 1.0
        return np.array([omega_max, delta_min]), grad

    gauss = np.exp(-(t - t_undress_begin) ** 2 / (2.0 * t_gaussian_width ** 2))
    omega = omega_max * gauss
    frac = (t - t_undress_begin) / t_gaussian_duration
    grad[1, 0] = frac
    grad[1, 1] = 1.0 - frac
    grad[1, 2] = -(delta_max - delta_min) * (t - t_mid - t_constant_duration / 2.0) / (n_gaussian_widths * t_gaussian_width ** 2)
    grad[1, 3] = -(delta_max - delta_min) / (2.0 * t_gaussian_duration)
    delta = delta_min + (delta_max - delta_min) * frac
    return np.array([omega, delta]), grad


def _two_photon_effective_controls_and_grad_fixed_omega(t, params, omega_1a_max, ratio=1.4):
    detuning_ratio, t_gaussian_width, t_constant_duration = params
    t_stop = t_constant_duration / 2.0
    t_abs = abs(t)
    grad = np.zeros((2, 3))

    if t_abs <= t_stop:
        omega_1a = omega_1a_max
        domega1 = np.array([0.0, 0.0, 0.0])
    else:
        gauss = np.exp(-(t_abs - t_stop) ** 2 / (2.0 * t_gaussian_width ** 2))
        omega_1a = omega_1a_max * gauss
        domega1 = np.array([
            0.0,
            omega_1a_max * gauss * (t_abs - t_stop) ** 2 / (t_gaussian_width ** 3),
            omega_1a_max * gauss * (t_abs - t_stop) / (2.0 * t_gaussian_width ** 2),
        ])

    omega_eff = ratio * omega_1a / (2.0 * detuning_ratio)
    delta_eff = (omega_1a ** 2 / omega_1a_max + omega_1a_max * ratio ** 2) / (4.0 * detuning_ratio)

    grad[0, 0] = -omega_eff / detuning_ratio
    grad[0, 1] = ratio * domega1[1] / (2.0 * detuning_ratio)
    grad[0, 2] = ratio * domega1[2] / (2.0 * detuning_ratio)

    grad[1, 0] = -delta_eff / detuning_ratio
    grad[1, 1] = omega_1a * domega1[1] / (2.0 * detuning_ratio * omega_1a_max)
    grad[1, 2] = omega_1a * domega1[2] / (2.0 * detuning_ratio * omega_1a_max)

    return np.array([omega_eff, delta_eff]), grad


def _scale_time_params(params, old_t, new_t, indices):
    scaled = np.array(params, dtype=float)
    if old_t <= 0.0:
        return scaled
    factor = new_t / old_t
    for idx in indices:
        scaled[idx] *= factor
    return scaled


def _evaluate_case(case, params, Tcontrol):
    h_params = {'regime': '1-photon', 'V_rr': case['V_rr']}
    prop_params = PropagatorParameters(
        Nsteps=case.get('nsteps_eval', 400),
        Tstep=Tcontrol / case.get('nsteps_eval', 400),
        Tcontrol=Tcontrol,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )

    current_params = np.array(params, dtype=float)

    def phi_ansatz(t, _h_params):
        t_rel = t - Tcontrol / 2.0
        if case['kind'] == '1p':
            return _one_photon_controls_and_grad_fixed_omega(
                t_rel, current_params, case['omega_max'], case['n_gaussian_widths'], case['t_mid']
            )[0]
        return _two_photon_effective_controls_and_grad_fixed_omega(
            t_rel, current_params, case['omega_1a_max'], ratio=case['ratio']
        )[0]

    U_ramp = propagator_ivp(phi_ansatz, prop_params)
    infid = _spin_echo_metrics(U_ramp, [np.zeros((9, 9), dtype=complex)])[0]
    return infid, U_ramp


def _make_objective(case, Tcontrol):
    h_params = {'regime': '1-photon', 'V_rr': case['V_rr']}
    prop_params = PropagatorParameters(
        Nsteps=case.get('nsteps_opt', 120),
        Tstep=Tcontrol / case.get('nsteps_opt', 120),
        Tcontrol=Tcontrol,
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )

    current_params = np.array(case['initial'], dtype=float)

    def phi_ansatz(t, _h_params):
        t_rel = t - Tcontrol / 2.0
        if case['kind'] == '1p':
            return _one_photon_controls_and_grad_fixed_omega(
                t_rel, current_params, case['omega_max'], case['n_gaussian_widths'], case['t_mid']
            )[0]
        return _two_photon_effective_controls_and_grad_fixed_omega(
            t_rel, current_params, case['omega_1a_max'], ratio=case['ratio']
        )[0]

    def goat_hamiltonian_grad(t, phi_val, _h_params, params_vec):
        t_rel = t - Tcontrol / 2.0
        if case['kind'] == '1p':
            _, dphi = _one_photon_controls_and_grad_fixed_omega(
                t_rel, current_params, case['omega_max'], case['n_gaussian_widths'], case['t_mid']
            )
        else:
            _, dphi = _two_photon_effective_controls_and_grad_fixed_omega(
                t_rel, current_params, case['omega_1a_max'], ratio=case['ratio']
            )
        dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, {'regime': '1-photon', 'V_rr': case['V_rr']})
        return [dphi[0, k] * dH_domega + dphi[1, k] * dH_ddelta for k in range(len(params_vec))]

    prop_params.hamiltonian_matrix_grad_func = goat_hamiltonian_grad

    def objective(params):
        nonlocal current_params
        current_params = np.array(params, dtype=float)
        U_ramp, V_grads = propagator_with_gradient_ivp(phi_ansatz, prop_params, current_params)
        infid, grad, _ = _spin_echo_metrics(U_ramp, V_grads)
        return infid, grad

    return objective, prop_params


def _optimize_fixed_time(case, Tcontrol, x0):
    objective, _ = _make_objective(case, Tcontrol)
    res = scipy.optimize.minimize(
        fun=lambda x: objective(x)[0],
        x0=np.array(x0, dtype=float),
        jac=lambda x: objective(x)[1],
        method='L-BFGS-B',
        bounds=case['bounds'],
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': case.get('maxiter', 250)},
    )
    params_opt = np.array(res.x, dtype=float)
    infid, U_ramp = _evaluate_case(case, params_opt, Tcontrol)
    return {
        'Tcontrol': float(Tcontrol),
        'params': params_opt,
        'result': res,
        'infidelity': float(infid),
        'feasible': bool(infid < THRESHOLD),
        'U_ramp': U_ramp,
    }


def search_shortest_time(case, t_start, t_min, t_max, shrink=0.7, grow=2.0, bisection_tol=1e-3):
    time_indices = case['time_param_indices']
    trace = []

    def run_guess(Tcontrol, guess):
        trial = _optimize_fixed_time(case, Tcontrol, guess)
        trace.append({'Tcontrol': trial['Tcontrol'], 'infidelity': trial['infidelity']})
        return trial

    t_current = float(t_start)
    trial = run_guess(t_current, case['initial'])

    if trial['feasible']:
        best = trial
        lo = None
        hi = trial
        while t_current * shrink >= t_min:
            t_next = t_current * shrink
            guess = _scale_time_params(best['params'], best['Tcontrol'], t_next, time_indices)
            next_trial = run_guess(t_next, guess)
            if next_trial['feasible']:
                best = next_trial
                hi = next_trial
                t_current = t_next
            else:
                lo = next_trial
                break
        if lo is None:
            return best, trace
    else:
        lo = trial
        while t_current * grow <= t_max:
            t_next = t_current * grow
            guess = _scale_time_params(trial['params'], trial['Tcontrol'], t_next, time_indices)
            next_trial = run_guess(t_next, guess)
            if next_trial['feasible']:
                hi = next_trial
                break
            lo = next_trial
            trial = next_trial
            t_current = t_next
        else:
            raise RuntimeError('Could not reach threshold within t_max')

    while hi['Tcontrol'] - lo['Tcontrol'] > bisection_tol:
        t_mid = 0.5 * (hi['Tcontrol'] + lo['Tcontrol'])
        guess = _scale_time_params(hi['params'], hi['Tcontrol'], t_mid, time_indices)
        mid_trial = run_guess(t_mid, guess)
        if mid_trial['feasible']:
            hi = mid_trial
        else:
            lo = mid_trial

    return hi, trace


def _format_params(params):
    return ', '.join(f'{p:.12g}' for p in params)


def _waveforms_1p(case, params, Tcontrol, npts=2000):
    t = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, npts)
    omega = np.zeros_like(t)
    delta = np.zeros_like(t)
    for i, ti in enumerate(t):
        omega[i], delta[i] = _one_photon_controls_and_grad_fixed_omega(
            ti, params, case['omega_max'], case['n_gaussian_widths'], case['t_mid']
        )[0]
    return t, {'omega_primary': omega, 'omega_secondary': delta}


def _waveforms_2p(case, params, Tcontrol, npts=2000):
    t = np.linspace(-Tcontrol / 2.0, Tcontrol / 2.0, npts)
    omega_1a = np.zeros_like(t)
    omega_ar = np.zeros_like(t)
    omega_eff = np.zeros_like(t)
    delta_eff = np.zeros_like(t)
    for i, ti in enumerate(t):
        phi, _ = _two_photon_effective_controls_and_grad_fixed_omega(ti, params, case['omega_1a_max'], ratio=case['ratio'])
        omega_eff[i], delta_eff[i] = phi
        if abs(ti) <= params[2] / 2.0:
            omega_1a[i] = case['omega_1a_max']
        else:
            omega_1a[i] = case['omega_1a_max'] * math.exp(-(abs(ti) - params[2] / 2.0) ** 2 / (2.0 * params[1] ** 2))
        omega_ar[i] = case['ratio'] * case['omega_1a_max']
    return t, {'omega_primary': omega_1a, 'omega_secondary': omega_ar, 'omega_eff': omega_eff, 'delta_eff': delta_eff}


def make_results(case, trial, trace):
    if case['kind'] == '1p':
        t, waves = _waveforms_1p(case, trial['params'], trial['Tcontrol'])
    else:
        t, waves = _waveforms_2p(case, trial['params'], trial['Tcontrol'])

    out = {
        'case': case,
        'Tcontrol': trial['Tcontrol'],
        'trace': trace,
        'initial_infidelity': trace[0]['infidelity'],
        'optimized_infidelity': trial['infidelity'],
        'optimized_params': trial['params'],
        'waveforms': {'t': t, **waves},
        'result': trial['result'],
    }
    return out


def write_results_md(results, outfile):
    case = results['case']
    lines = ['# GOAT Spin-Echo Blockade Results', '']
    lines.append(f"## {case['label']}")
    lines.append(f"`V_rr = 2*pi`, `tau = t / (2*pi / V_rr)`")
    lines.append(f"Initial infidelity: `{results['initial_infidelity']:.6e}`")
    lines.append(f"Optimized infidelity: `{results['optimized_infidelity']:.6e}`")
    lines.append(f"Shortest feasible `Tbar`: `{results['Tcontrol']:.12g}`")
    lines.append('')
    lines.append('| Parameter | Value |')
    lines.append('| --- | ---: |')
    if case['kind'] == '1p':
        names = ['Delta_max / V_rr', 'Delta_min / V_rr', 't_gaussian_width * V_rr / (2*pi)', 't_constant_duration * V_rr / (2*pi)']
    else:
        names = ['Delta_1a / Omega_1a_max', 't_gaussian_width * V_rr / (2*pi)', 't_constant_duration * V_rr / (2*pi)']
    for name, value in zip(names, results['optimized_params']):
        lines.append(f'| `{name}` | `{value:.12g}` |')
    lines.append('')
    lines.append('### Search Trace')
    lines.append('| Tbar | Infidelity |')
    lines.append('| ---: | ---: |')
    for item in trace_rows(results['trace']):
        lines.append(f"| `{item[0]:.12g}` | `{item[1]:.6e}` |")
    Path(outfile).write_text('\n'.join(lines) + '\n')


def trace_rows(trace):
    return [(row['Tcontrol'], row['infidelity']) for row in trace]


def write_results_h5(results, outfile):
    if h5py is None:
        raise ModuleNotFoundError('h5py is required to write HDF5 results')

    case = results['case']
    with h5py.File(outfile, 'w') as h5:
        h5.attrs['V_rr'] = V_RR
        h5.attrs['time_unit'] = TIME_UNIT
        h5.attrs['threshold'] = THRESHOLD
        grp = h5.create_group(case['name'])
        grp.attrs['label'] = case['label']
        grp.attrs['kind'] = case['kind']
        grp.attrs['omega_max'] = case['omega_max'] if case['kind'] == '1p' else case['omega_1a_max']
        grp.attrs['Tcontrol'] = results['Tcontrol']
        grp.attrs['initial_infidelity'] = results['initial_infidelity']
        grp.attrs['optimized_infidelity'] = results['optimized_infidelity']
        grp.create_dataset('optimized_params', data=results['optimized_params'])
        grp.create_dataset('t', data=results['waveforms']['t'])
        for key, value in results['waveforms'].items():
            if key != 't':
                grp.create_dataset(key, data=value)
        grp.create_dataset('trace_Tbar', data=np.array([row['Tcontrol'] for row in results['trace']], dtype=float))
        grp.create_dataset('trace_infidelity', data=np.array([row['infidelity'] for row in results['trace']], dtype=float))


def make_1p_case():
    return {
        'name': 'blockade_1p',
        'label': '1-photon strong blockade',
        'kind': '1p',
        'V_rr': V_RR,
        'omega_max': 0.1 * V_RR,
        'n_gaussian_widths': 4,
        't_mid': 0.0,
        'initial': np.array([
            -20.0,
            -3.0,
            0.1909859317102744,
            1.7435356680443814,
        ], dtype=float),
        'bounds': [
            (-50.0, 0.0),
            (-20.0, 0.0),
            (0.01, 50.0),
            (0.0, 100.0),
        ],
        'time_param_indices': [2, 3],
        'nsteps_opt': 80,
        'nsteps_eval': 240,
        'maxiter': 120,
    }


def make_2p_case():
    return {
        'name': 'blockade_2p',
        'label': '2-photon strong blockade',
        'kind': '2p',
        'V_rr': V_RR,
        'omega_1a_max': 0.1 * V_RR,
        'ratio': 1.4,
        'initial': np.array([
            37.699111843,
            0.184120046,
            0.5977988994497249,
        ], dtype=float),
        'bounds': [
            (1.0, 100.0),
            (0.01, 50.0),
            (0.0, 100.0),
        ],
        'time_param_indices': [1, 2],
        'nsteps_opt': 80,
        'nsteps_eval': 240,
        'maxiter': 120,
    }


def run_case(case, t_start, t_min=0.1, t_max=200.0, nsteps_opt=None, nsteps_eval=None, maxiter=None):
    case = dict(case)
    if nsteps_opt is not None:
        case['nsteps_opt'] = nsteps_opt
    if nsteps_eval is not None:
        case['nsteps_eval'] = nsteps_eval
    if maxiter is not None:
        case['maxiter'] = maxiter

    trial, trace = search_shortest_time(case, t_start=t_start, t_min=t_min, t_max=t_max)
    if not trial['feasible']:
        raise RuntimeError(f"{case['name']} did not reach threshold")
    return make_results(case, trial, trace)
