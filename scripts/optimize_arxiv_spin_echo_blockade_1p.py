import argparse

from goat_spin_echo_blockade import make_1p_case, run_case, write_results_h5, write_results_md


def main():
    parser = argparse.ArgumentParser(description='Optimize 1-photon blockade MS spin-echo passage')
    parser.add_argument('--t-start', type=float, default=32.7142419188588, help='Initial dimensionless Tbar')
    parser.add_argument('--t-min', type=float, default=0.25, help='Minimum dimensionless Tbar to explore')
    parser.add_argument('--t-max', type=float, default=500.0, help='Maximum dimensionless Tbar to explore')
    parser.add_argument('--nsteps-opt', type=int, default=80, help='Optimization integration steps')
    parser.add_argument('--nsteps-eval', type=int, default=240, help='Evaluation integration steps')
    parser.add_argument('--maxiter', type=int, default=120, help='Optimizer iteration cap')
    parser.add_argument('--md-out', type=str, default='results_blockade_1p.md', help='Markdown output file')
    parser.add_argument('--h5-out', type=str, default='results_blockade_1p.h5', help='HDF5 output file')
    args = parser.parse_args()

    case = make_1p_case()
    result = run_case(
        case,
        t_start=args.t_start,
        t_min=args.t_min,
        t_max=args.t_max,
        nsteps_opt=args.nsteps_opt,
        nsteps_eval=args.nsteps_eval,
        maxiter=args.maxiter,
    )
    write_results_md(result, args.md_out)
    write_results_h5(result, args.h5_out)
    print(f"{case['name']} Tbar={result['Tcontrol']:.12g} infidelity={result['optimized_infidelity']:.6e}")


if __name__ == '__main__':
    main()
