import math
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


QUBIT_IDXS = [0, 1, 3, 4]
M_PI_2 = microwave_x_pulse_9d(np.pi / 2.0)
M_PI = microwave_x_pulse_9d(np.pi)
TARGET_4X4 = get_ms_yy_target_4x4()

CURRENT_CASE = None
CURRENT_PARAMS = None


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

    return infidelity, grad, U_echo


def _one_photon_controls_and_grad(t, params, n_gaussian_widths, t_mid=0.0):
    omega_max, delta_max, delta_min, t_gaussian_width, t_constant_duration = params
    t_gaussian_duration = n_gaussian_widths * t_gaussian_width

    t_dress_begin = t_mid - t_gaussian_duration - t_constant_duration / 2.0
    t_dress_end = t_mid - t_constant_duration / 2.0
    t_undress_begin = t_mid + t_constant_duration / 2.0
    t_undress_end = t_mid + t_gaussian_duration + t_constant_duration / 2.0

    grad = np.zeros((2, 5))

    if t < t_dress_begin or t > t_undress_end:
        grad[1, 1] = 1.0
        return np.array([0.0, delta_max]), grad

    if t < t_dress_end:
        gauss = np.exp(-(t - t_dress_end) ** 2 / (2.0 * t_gaussian_width ** 2))
        omega = omega_max * gauss
        grad[0, 0] = gauss
        grad[0, 3] = omega_max * gauss * (t - t_dress_end) ** 2 / (t_gaussian_width ** 3)
        grad[0, 4] = -omega_max * gauss * (t - t_dress_end) / (2.0 * t_gaussian_width ** 2)

        denom = t_gaussian_duration
        frac = (t - t_dress_begin) / denom
        delta = delta_max + (delta_min - delta_max) * frac
        a = t - t_mid + t_constant_duration / 2.0
        grad[1, 1] = 1.0 - frac
        grad[1, 2] = frac
        grad[1, 3] = -(delta_min - delta_max) * a / (n_gaussian_widths * t_gaussian_width ** 2)
        grad[1, 4] = (delta_min - delta_max) / (2.0 * denom)
        return np.array([omega, delta]), grad

    if t <= t_undress_begin:
        grad[0, 0] = 1.0
        grad[1, 2] = 1.0
        return np.array([omega_max, delta_min]), grad

    gauss = np.exp(-(t - t_undress_begin) ** 2 / (2.0 * t_gaussian_width ** 2))
    omega = omega_max * gauss
    grad[0, 0] = gauss
    grad[0, 3] = omega_max * gauss * (t - t_undress_begin) ** 2 / (t_gaussian_width ** 3)
    grad[0, 4] = omega_max * gauss * (t - t_undress_begin) / (2.0 * t_gaussian_width ** 2)

    denom = t_gaussian_duration
    frac = (t - t_undress_begin) / denom
    delta = delta_min + (delta_max - delta_min) * frac
    b = t - t_mid - t_constant_duration / 2.0
    grad[1, 1] = frac
    grad[1, 2] = 1.0 - frac
    grad[1, 3] = -(delta_max - delta_min) * b / (n_gaussian_widths * t_gaussian_width ** 2)
    grad[1, 4] = -(delta_max - delta_min) / (2.0 * denom)
    return np.array([omega, delta]), grad


def _two_photon_effective_controls_and_grad(t, params, ratio=1.4):
    omega_1a_max, detuning_ratio, t_gaussian_width, t_constant_duration = params
    t_stop = t_constant_duration / 2.0
    t_abs = abs(t)
    grad = np.zeros((2, 4))

    if t_abs <= t_stop:
        omega_1a = omega_1a_max
        domega1 = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        gauss = np.exp(-(t_abs - t_stop) ** 2 / (2.0 * t_gaussian_width ** 2))
        omega_1a = omega_1a_max * gauss
        domega1 = np.array([
            gauss,
            0.0,
            omega_1a_max * gauss * (t_abs - t_stop) ** 2 / (t_gaussian_width ** 3),
            omega_1a_max * gauss * (t_abs - t_stop) / (2.0 * t_gaussian_width ** 2),
        ])

    omega_eff = ratio * omega_1a_max * (omega_1a / omega_1a_max) / (2.0 * detuning_ratio)
    delta_eff = omega_1a_max * (((omega_1a / omega_1a_max) ** 2) + ratio ** 2) / (4.0 * detuning_ratio)

    grad[0, 0] = ratio * (omega_1a / omega_1a_max) / (2.0 * detuning_ratio)
    grad[0, 1] = -omega_eff / detuning_ratio
    grad[0, 2] = ratio * omega_1a_max * domega1[2] / (2.0 * detuning_ratio)
    grad[0, 3] = ratio * omega_1a_max * domega1[3] / (2.0 * detuning_ratio)

    grad[1, 0] = (((omega_1a / omega_1a_max) ** 2) + ratio ** 2) / (4.0 * detuning_ratio)
    grad[1, 1] = -delta_eff / detuning_ratio
    grad[1, 2] = omega_1a_max * (omega_1a / omega_1a_max) * (domega1[2] / omega_1a_max) / (2.0 * detuning_ratio)
    grad[1, 3] = omega_1a_max * (omega_1a / omega_1a_max) * (domega1[3] / omega_1a_max) / (2.0 * detuning_ratio)

    return np.array([omega_eff, delta_eff]), grad


def _objective_factory(case):
    global CURRENT_CASE, CURRENT_PARAMS
    CURRENT_CASE = case

    h_params = {'regime': '1-photon', 'V_rr': case['V_rr']}
    prop_params = PropagatorParameters(
        Nsteps=100,
        Tstep=case['Tcontrol'] / 100,
        Tcontrol=case['Tcontrol'],
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )

    def phi_ansatz(t, _h_params):
        t_rel = t - case['Tcontrol'] / 2.0
        CURRENT_PARAMS = CURRENT_CASE['current_params']
        if case['kind'] == '1p':
            return _one_photon_controls_and_grad(t_rel, CURRENT_PARAMS, case['n_gaussian_widths'], case['t_mid'])[0]
        return _two_photon_effective_controls_and_grad(t_rel, CURRENT_PARAMS, ratio=case['ratio'])[0]

    def goat_hamiltonian_grad(t, phi_val, _h_params, params_vec):
        t_rel = t - case['Tcontrol'] / 2.0
        CURRENT_PARAMS = CURRENT_CASE['current_params']
        if case['kind'] == '1p':
            _, dphi = _one_photon_controls_and_grad(t_rel, CURRENT_PARAMS, case['n_gaussian_widths'], case['t_mid'])
        else:
            _, dphi = _two_photon_effective_controls_and_grad(t_rel, CURRENT_PARAMS, ratio=case['ratio'])
        dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, {'regime': '1-photon', 'V_rr': case['V_rr']})
        return [dphi[0, k] * dH_domega + dphi[1, k] * dH_ddelta for k in range(len(params_vec))]

    prop_params.hamiltonian_matrix_grad_func = goat_hamiltonian_grad

    def objective(params):
        CURRENT_CASE['current_params'] = np.array(params, dtype=float)
        U_ramp, V_grads = propagator_with_gradient_ivp(phi_ansatz, prop_params, params)
        infid, grad, _ = _spin_echo_metrics(U_ramp, V_grads)
        return infid, grad

    return objective, prop_params


def _evaluate_case(case, params):
    h_params = {'regime': '1-photon', 'V_rr': case['V_rr']}
    prop_params = PropagatorParameters(
        Nsteps=200,
        Tstep=case['Tcontrol'] / 200,
        Tcontrol=case['Tcontrol'],
        hamiltonian_params=h_params,
        hamiltonian_matrix_func=hamiltonian_ms_gate,
        hamiltonian_matrix_grad_func=None,
    )

    def phi_ansatz(t, _h_params):
        t_rel = t - case['Tcontrol'] / 2.0
        if case['kind'] == '1p':
            return _one_photon_controls_and_grad(t_rel, params, case['n_gaussian_widths'], case['t_mid'])[0]
        return _two_photon_effective_controls_and_grad(t_rel, params, ratio=case['ratio'])[0]

    U_ramp = propagator_ivp(phi_ansatz, prop_params)
    infid = _spin_echo_metrics(U_ramp, [np.zeros((9, 9), dtype=complex)])[0]
    return infid, U_ramp


def _waveforms(case, params):
    t = np.linspace(-case['Tcontrol'] / 2.0, case['Tcontrol'] / 2.0, 2000)
    if case['kind'] == '1p':
        omega = np.zeros_like(t)
        delta = np.zeros_like(t)
        for i, ti in enumerate(t):
            (omega[i], delta[i]), _ = _one_photon_controls_and_grad(ti, params, case['n_gaussian_widths'], case['t_mid'])
        return t, {'omega': omega, 'delta': delta}

    omega_1a = np.zeros_like(t)
    omega_ar = np.zeros_like(t)
    omega_eff = np.zeros_like(t)
    delta_eff = np.zeros_like(t)
    for i, ti in enumerate(t):
        phi, _ = _two_photon_effective_controls_and_grad(ti, params, ratio=case['ratio'])
        omega_eff[i], delta_eff[i] = phi
        omega_1a[i] = _one_photon_like_two_photon_ramp(ti, params)
        omega_ar[i] = case['ratio'] * params[0]
    return t, {'omega_1a': omega_1a, 'omega_ar': omega_ar, 'omega_eff': omega_eff, 'delta_eff': delta_eff}


def _one_photon_like_two_photon_ramp(t, params):
    omega_1a_max, _detuning_ratio, t_gaussian_width, t_constant_duration = params
    t_stop = t_constant_duration / 2.0
    if abs(t) <= t_stop:
        return omega_1a_max
    return omega_1a_max * math.exp(-(abs(t) - t_stop) ** 2 / (2.0 * t_gaussian_width ** 2))


def _format_params(params):
    return ", ".join(f"{p:.12g}" for p in params)


def main():
    cases = [
        {
            'name': '1p_vanilla',
            'label': '1-photon vanilla',
            'kind': '1p',
            'V_rr': 10.0 * (2 * np.pi),
            'Tcontrol': 8.0703458766310,
            'n_gaussian_widths': 2,
            't_mid': 0.0,
            'initial': np.array([
                1.0 * (2 * np.pi),
                -5.0 * (2 * np.pi),
                -0.16 * (2 * np.pi),
                1.909859317102744,
                0.430928242225502,
            ], dtype=float),
            'bounds': [
                (0.1 * (2 * np.pi), 10.0 * (2 * np.pi)),
                (-20.0 * (2 * np.pi), 0.0),
                (-5.0 * (2 * np.pi), 0.0),
                (0.05, 10.0),
                (0.0, 20.0),
            ],
        },
        {
            'name': '1p_strong',
            'label': '1-photon strong blockade',
            'kind': '1p',
            'V_rr': 0.1 * (2 * np.pi),
            'Tcontrol': 32.7142419188588,
            'n_gaussian_widths': 4,
            't_mid': 0.0,
            'initial': np.array([
                1.0 * (2 * np.pi),
                -2.0 * (2 * np.pi),
                -0.3 * (2 * np.pi),
                1.909859317102744,
                17.435356680443814,
            ], dtype=float),
            'bounds': [
                (0.1 * (2 * np.pi), 10.0 * (2 * np.pi)),
                (-20.0 * (2 * np.pi), 0.0),
                (-5.0 * (2 * np.pi), 0.0),
                (0.05, 10.0),
                (0.0, 40.0),
            ],
        },
        {
            'name': '2p_strong',
            'label': '2-photon strong blockade',
            'kind': '2p',
            'V_rr': 0.1 * (2 * np.pi),
            'Tcontrol': 20.0,
            'ratio': 1.4,
            'initial': np.array([
                100.0 * (2 * np.pi),
                37.699111843,
                1.84120046,
                5.977988994497249,
            ], dtype=float),
            'bounds': [
                (10.0 * (2 * np.pi), 1000.0 * (2 * np.pi)),
                (1.0, 100.0),
                (0.05, 10.0),
                (0.0, 20.0),
            ],
        },
        {
            'name': '2p_weak',
            'label': '2-photon weak blockade',
            'kind': '2p',
            'V_rr': 0.1 * (2 * np.pi),
            'Tcontrol': 20.0,
            'ratio': 1.4,
            'initial': np.array([
                100.0 * (2 * np.pi),
                20.0,
                0.646265625,
                0.0,
            ], dtype=float),
            'bounds': [
                (10.0 * (2 * np.pi), 1000.0 * (2 * np.pi)),
                (1.0, 100.0),
                (0.05, 10.0),
                (0.0, 20.0),
            ],
        },
    ]

    results = []
    for case in cases:
        objective, _ = _objective_factory(case)
        case['current_params'] = case['initial'].copy()
        infid0, _ = objective(case['initial'])
        res = scipy.optimize.minimize(
            fun=lambda x: objective(x)[0],
            x0=case['initial'],
            jac=lambda x: objective(x)[1],
            method='L-BFGS-B',
            bounds=case['bounds'],
            options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 200},
        )
        params_opt = np.array(res.x, dtype=float)
        case['current_params'] = params_opt.copy()
        infid, _ = objective(params_opt)
        if case['kind'] == '1p':
            t, waves = _waveforms(case, params_opt)
            omega_ref = params_opt[0]
            wave_out = {
                't': t,
                'omega_primary': waves['omega'],
                'omega_secondary': waves['delta'],
                'omega_ref': omega_ref,
            }
        else:
            t, waves = _waveforms(case, params_opt)
            omega_ref = params_opt[0]
            wave_out = {
                't': t,
                'omega_primary': waves['omega_1a'],
                'omega_secondary': waves['omega_ar'],
                'omega_eff': waves['omega_eff'],
                'delta_eff': waves['delta_eff'],
                'omega_ref': omega_ref,
            }

        results.append({
            'case': case,
            'initial_infidelity': float(infid0),
            'optimized_infidelity': float(infid),
            'opt_result': res,
            'optimized_params': params_opt,
            'waveforms': wave_out,
        })

    _write_results_md(results)
    _write_results_h5(results)

    for item in results:
        print(item['case']['name'], item['optimized_infidelity'], _format_params(item['optimized_params']))


def _write_results_md(results):
    lines = ['# GOAT Spin-Echo Optimization Results', '']
    for item in results:
        case = item['case']
        lines.append(f"## {case['label']}")
        lines.append(f"Initial infidelity: `{item['initial_infidelity']:.6e}`")
        lines.append(f"Optimized infidelity: `{item['optimized_infidelity']:.6e}`")
        lines.append('')
        lines.append('| Parameter | Value |')
        lines.append('| --- | ---: |')
        if case['kind'] == '1p':
            names = ['Omega_max', 'Delta_max', 'Delta_min', 't_gaussian_width', 't_constant_duration']
        else:
            names = ['Omega_1a_max', 'Delta_1a/Omega_1a_max', 't_gaussian_width', 't_constant_duration']
        for n, v in zip(names, item['optimized_params']):
            lines.append(f'| `{n}` | `{v:.12g}` |')
        lines.append('')
    Path('results.md').write_text('\n'.join(lines) + '\n')


def _write_results_h5(results):
    with h5py.File('results.h5', 'w') as h5:
        h5.attrs['source_params_tex'] = 'arxiv_paper/params.tex'
        h5.attrs['spin_echo_target'] = 'MS_yy'
        for item in results:
            case = item['case']
            grp = h5.create_group(case['name'])
            grp.attrs['label'] = case['label']
            grp.attrs['kind'] = case['kind']
            grp.attrs['V_rr'] = case['V_rr']
            grp.attrs['Tcontrol'] = case['Tcontrol']
            grp.attrs['initial_infidelity'] = item['initial_infidelity']
            grp.attrs['optimized_infidelity'] = item['optimized_infidelity']
            grp.create_dataset('initial_params', data=case['initial'])
            grp.create_dataset('optimized_params', data=item['optimized_params'])
            for key, value in item['waveforms'].items():
                grp.create_dataset(key, data=value)


if __name__ == '__main__':
    main()
