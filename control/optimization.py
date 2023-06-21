import numpy as np
import scipy.optimize

from numpy import pi

from params import ControlProblem


def grape(control_problem, debug=False, gtol=1e-5, maxiter=None):
    if isinstance(control_problem, dict):
        control_problem = ControlProblem.from_dict(control_problem)

    cost_function = control_problem.cost_function
    cost_function_grad = control_problem.cost_function_grad
    Nsteps = control_problem.propagator_params.Nsteps

    initialization = control_problem.initialization if control_problem.initialization != 'Random' else 'Random'
    if initialization == 'Random':
        phi_initial = 2 * pi * np.random.rand(Nsteps)
    elif initialization == 'Constant':
        phi_initial = np.zeros(Nsteps)
    elif initialization == 'Sine':
        phi_initial = pi / 2 * (1 + np.sin(2 * pi * np.linspace(0, 2, Nsteps)))

    jac = True if cost_function_grad is None else cost_function_grad
    result = scipy.optimize.minimize(
        fun=cost_function,
        x0=phi_initial,
        jac=jac,
        method='BFGS',
        options={'gtol': gtol, 'maxiter': maxiter},
        args=(control_problem,),
    )

    phi_des = result.x
    infidelity_min = result.fun
    dinfidelity_min = result.jac
    Niterations = result.nit

    if debug:
        print('# dinfidelity_min = %s\n # infidelity_min = %g\n # Niterations = %d' % (dinfidelity_min, infidelity_min, Niterations))

    control_problem.results = {
        'PhiInitial': phi_initial,
        'PhiOptimized': phi_des,
        'InfidelityMin': infidelity_min,
        'InfidelityMinGradient': dinfidelity_min,
        'Niterations': Niterations,
    }
    return phi_des, infidelity_min
