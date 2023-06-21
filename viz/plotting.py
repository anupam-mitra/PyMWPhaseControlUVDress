import numpy as np

from dataio.io_utils import load_cases_from_configs, load_results_cases


def plot_one_photon_cases(cases, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 3.2 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        t = case['t']
        ax.plot(t, case['omega_primary'] / (2.0 * np.pi), label=r'$\Omega_{1r} / 2\pi$', color='tab:blue')
        ax.plot(t, case['omega_secondary'] / (2.0 * np.pi), label=r'$\Delta_{1r} / 2\pi$', color='tab:orange')
        ax.set_title(f"{case['label']} (GOAT $1-F$ = {case['infidelity']:.2e})")
        ax.set_xlabel('t [us]')
        ax.set_ylabel('MHz')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')


def plot_two_photon_cases(cases, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 3.2 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        t = case['t']
        ax.plot(t, case['omega_primary'] / (2.0 * np.pi), label=r'$\Omega_{1a} / 2\pi$', color='tab:blue')
        ax.plot(t, case['omega_secondary'] / (2.0 * np.pi), label=r'$\Omega_{ar} / 2\pi$', color='tab:orange', linestyle='--')
        ax.set_title(f"{case['label']} (GOAT $1-F$ = {case['infidelity']:.2e})")
        ax.set_xlabel('t [us]')
        ax.set_ylabel('MHz')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')


def plot_arxiv_paper_pulses(results_path='results.h5', out_one='arxiv_1p_pulses.pdf', out_two='arxiv_2p_pulses.pdf'):
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        h5py = None

    if results_path and h5py is not None:
        try:
            one_photon_cases, two_photon_cases = load_results_cases(results_path)
        except FileNotFoundError:
            one_photon_cases, two_photon_cases = load_cases_from_configs()
    else:
        one_photon_cases, two_photon_cases = load_cases_from_configs()

    plot_one_photon_cases(one_photon_cases, out_one)
    plot_two_photon_cases(two_photon_cases, out_two)
