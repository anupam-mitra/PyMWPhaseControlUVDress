import numpy as np

from core.unitaries import (
    evaluate_one_photon_ms_infidelity,
    evaluate_one_photon_ms_spin_echo_infidelity,
    evaluate_two_photon_ms_infidelity,
    evaluate_two_photon_ms_spin_echo_infidelity,
    get_ms_yy_target,
)


def report(name, infid):
    print(f"{name}: infidelity={infid:.6f} {'OK' if infid < 1e-3 else 'NO'}")


def main():
    target_u = get_ms_yy_target()

    one_photon_cases = [
        (
            "1p vanilla",
            {
                'regime': '1-photon',
                'V_rr': 10.0 * (2 * np.pi),
            },
            {
                'Omega_max': 1.0 * (2 * np.pi),
                'Omega_min': 0.0 * (2 * np.pi),
                'Delta_max': -5.0 * (2 * np.pi),
                'Delta_min': -0.16 * (2 * np.pi),
                't_gaussian_width': 1.909859317102744,
                't_constant_duration': 0.430928242225502,
                't_mid': 0.0,
                'n_gaussian_widths': 2,
                'Tcontrol': 8.0703458766310,
            },
        ),
        (
            "1p strong",
            {
                'regime': '1-photon',
                'V_rr': 0.1 * (2 * np.pi),
            },
            {
                'Omega_max': 1.0 * (2 * np.pi),
                'Omega_min': 0.0 * (2 * np.pi),
                'Delta_max': -2.0 * (2 * np.pi),
                'Delta_min': -0.3 * (2 * np.pi),
                't_gaussian_width': 1.909859317102744,
                't_constant_duration': 17.435356680443814,
                't_mid': 0.0,
                'n_gaussian_widths': 4,
                'Tcontrol': 32.7142419188588,
            },
        ),
    ]

    two_photon_cases = [
        (
            "2p strong blockade",
            {
                'regime': '2-photon',
                'V_rr': 0.1 * (2 * np.pi),
                'Omega_ar': 1.4 * 100 * (2 * np.pi),
                'Delta_1a': 37.699111843 * (100 * (2 * np.pi)),
                'Delta_ar': -37.699111843 * (100 * (2 * np.pi)),
                'Delta_max': 0.0,
            },
            {
                'Omega_1a_max': 100 * (2 * np.pi),
                't_constant_duration': 5.977988994497249,
                't_gaussian_width': 1.84120046,
                'Tcontrol': 20.0,
            },
        ),
        (
            "2p weak blockade",
            {
                'regime': '2-photon',
                'V_rr': 0.1 * (2 * np.pi),
                'Omega_ar': 1.4 * 100 * (2 * np.pi),
                'Delta_1a': 20.0 * (100 * (2 * np.pi)),
                'Delta_ar': -20.0 * (100 * (2 * np.pi)),
                'Delta_max': 0.0,
            },
            {
                'Omega_1a_max': 100 * (2 * np.pi),
                't_constant_duration': 0.0,
                't_gaussian_width': 0.646265625,
                'Tcontrol': 20.0,
            },
        ),
    ]

    for name, h_params, pulse_params in one_photon_cases:
        infid = evaluate_one_photon_ms_infidelity(pulse_params, h_params, target_u=target_u)
        spin_echo_infid, _ = evaluate_one_photon_ms_spin_echo_infidelity(pulse_params, h_params)
        report(f"{name} raw", infid)
        report(f"{name} spin-echo", spin_echo_infid)

    for name, h_params, pulse_params in two_photon_cases:
        infid = evaluate_two_photon_ms_infidelity(pulse_params, h_params, target_u=target_u)
        spin_echo_infid, _ = evaluate_two_photon_ms_spin_echo_infidelity(pulse_params, h_params)
        report(f"{name} raw", infid)
        report(f"{name} spin-echo", spin_echo_infid)


if __name__ == "__main__":
    main()
