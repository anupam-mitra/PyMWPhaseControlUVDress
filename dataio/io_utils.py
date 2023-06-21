from pathlib import Path

import numpy as np

from core.unitaries import one_photon_ramp, two_photon_omega_ramp, _two_photon_t_stop


CONFIG_DIR = Path(__file__).resolve().parent / 'configs'


def load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def label_from_config_path(path):
    stem = Path(path).stem
    parts = stem.replace('ms_', '').replace('_blockade', '').split('_')
    if len(parts) >= 2:
        return f"{parts[0][0]}-photon {parts[1]} blockade"
    return stem


def load_cases_from_configs():
    one_photon_specs = ['ms_1p_strong_blockade.yaml', 'ms_1p_weak_blockade.yaml']
    two_photon_specs = ['ms_2p_strong_blockade.yaml', 'ms_2p_weak_blockade.yaml']

    one_photon_cases = []
    for filename in one_photon_specs:
        config = load_yaml(CONFIG_DIR / filename)
        h_params = config['hamiltonian_params']
        c_params = config['control_params']
        label = label_from_config_path(filename)
        t = np.linspace(-c_params['Tcontrol'] / 2.0, c_params['Tcontrol'] / 2.0, 2000)
        omega = np.zeros_like(t)
        delta = np.zeros_like(t)
        for i, ti in enumerate(t):
            omega[i], delta[i] = one_photon_ramp(ti, {**h_params, **c_params, 'Tcontrol': c_params['Tcontrol']})
        one_photon_cases.append({
            'label': label,
            'infidelity': np.nan,
            't': t,
            'omega_primary': omega,
            'omega_secondary': delta,
        })

    two_photon_cases = []
    for filename in two_photon_specs:
        config = load_yaml(CONFIG_DIR / filename)
        h_params = config['hamiltonian_params']
        c_params = config['control_params']
        label = label_from_config_path(filename)
        t = np.linspace(-c_params['Tcontrol'] / 2.0, c_params['Tcontrol'] / 2.0, 2000)
        omega_1a = np.array([two_photon_omega_ramp(ti, h_params['Omega_1a_max'], _two_photon_t_stop(c_params), h_params['t_gaussian_width']) for ti in t])
        omega_ar = np.full_like(t, h_params['Omega_ar'])
        two_photon_cases.append({
            'label': label,
            'infidelity': np.nan,
            't': t,
            'omega_primary': omega_1a,
            'omega_secondary': omega_ar,
        })

    return one_photon_cases, two_photon_cases


def load_results_cases(path):
    import h5py

    preferred = {'1p_vanilla': 0, '1p_strong': 1, '2p_strong': 2, '2p_weak': 3}
    one_photon_cases = []
    two_photon_cases = []

    with h5py.File(path, 'r') as h5:
        for name in sorted(h5.keys(), key=lambda n: preferred.get(n, 999)):
            grp = h5[name]
            item = {
                'name': name,
                'label': grp.attrs['label'].decode() if isinstance(grp.attrs['label'], bytes) else str(grp.attrs['label']),
                'kind': grp.attrs['kind'].decode() if isinstance(grp.attrs['kind'], bytes) else str(grp.attrs['kind']),
                'infidelity': float(grp.attrs['optimized_infidelity']),
                't': np.array(grp['t']),
                'omega_primary': np.array(grp['omega_primary']),
                'omega_secondary': np.array(grp['omega_secondary']),
            }
            if item['kind'] == '1p':
                one_photon_cases.append(item)
            else:
                two_photon_cases.append(item)
    return one_photon_cases, two_photon_cases


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
    for row in results['trace']:
        lines.append(f"| `{row['Tcontrol']:.12g}` | `{row['infidelity']:.6e}` |")
    Path(outfile).write_text('\n'.join(lines) + '\n')


def write_results_h5(results, outfile, *, attrs=None):
    import h5py

    case = results['case']
    with h5py.File(outfile, 'w') as h5:
        if attrs:
            for key, value in attrs.items():
                h5.attrs[key] = value
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
