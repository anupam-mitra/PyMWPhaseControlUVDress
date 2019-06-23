#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  grape.py
#
#  Copyright 2017 Anupam Mitra <anupam@unm.edu>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

from __future__ import division

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize

from numpy import pi

def initialize_guess (initialization, nsteps, ncontrols, tsteps, \
    control_variables_min, control_variables_max):
    """
    Makes an initial guess for the time dependence of a control field for
    quantum optimal control

    Parameters
    ----------
    initialization: str
    Method to use for initialization

    nsteps: int
    Number of time steps for the initial guess

    ncontrols: int
    Number of control fields

    tsteps: ndarray <real> (nsteps)
    Duration of time steps

    control_variables_min: ndarray <real> (ncontrols)
    Minimum value of each control variable

    control_variables_max: ndarray <real> (ncontrols)
    Maximum value of each control variable

    Returns
    -------
    control_variables_initial: ndarray <real> (nsteps, ncontrols)
    Initial guess for the control variables
    """

    tcontrol = np.sum(tsteps)
    control_variables_initial = np.empty((nsteps, ncontrols))

    for c in range(ncontrols):
        control_variable_range = \
            (control_variables_max[c] - control_variables_min[c])

        if initialization.lower() == 'random':
            control_variables_initial[:, c] = control_variables_min[c] + \
                 control_variable_range * np.random.randn((nsteps, ncontrols))

        elif initialization.lower() == 'constant':
            control_variables_initial = control_variable_min[c] +
                 control_variable_range/2

        elif  initialization.lower() == 'sine':
            control_variables_initial[:, c] = control_variable_min[c] +
                 control_variable_range * (1 - np.cos(pi / tcontrol * tsteps))/2

        elif initialization.lower() == 'linear':
            control_variables_initial[:, c] = control_variable_min[c] +
                 control_variable_range / tcontrol * tsteps

    return control_variables_initial


def costfunction_infidelity_wrap (control_variables_flattened, control_problem):
    """
    Computes the cost function for a particular control sequence, specified
    in the time domain for quantum optimal control. This is wrapper around the
    function `infidelity`. This function extracts the parameters from a
    dictionary `control_problem` and passes them to `infidelity`. It also
    reshapes the `control_variables` to the appropriate size

    The basic procedure is as follows.

    Compute the Hamiltonian as a function of time and its derivatives as a
    function of time.

    Compute the propagators and its derivatives

    Compute the infidelity cost function and it derivatives from the propagator
    and its derivatives

    The "arguments" over which the cost function is minimized are the control
    parameters. The other parameters like static parameters of the Hamiltonian,
    control time, number of steps, target, etc are not optimized over.

    Minimization functions provided by scipy expect the function to be minimized
    to have an argument which is an array of numbers representing the arguments
    over which the function is minimized.

    """
    nsteps = control_problem['costfunction/propagators/nsteps']
    ncontrols = control_problem['costfunction/propagators/ncontrols']

    costfunction = control_problem['costfunction/costfunction']
    full_unitary_function = \
        control_problem.get('costfunction/propagators/full_unitary_function')

    control_variables =  np.reshape(control_variables_flattened, \
                                    (ncontrols, nsteps))

    infidelity_value, grad_infidelity_value = \
        costfunction(control_variables, static_parameters, target, ndim, \
            hamiltonian_function, tsteps, weights, \
            full_unitary_function)

    return infidelity_value, grad_infidelity_value

def stepwise_constant_quantum_control (control_problem):
    """
    Optimizes a step wise constant set of control fields for quantum optimal
    control

    Parameters
    -----------
    control_problrem: dictionary
    Contains the different parameters for a quantum optimal control
    problem

    Reads the following keys for the cost function calculation

    "costfunction/costfunction": function
    Python function which calculates the cost function and its gradient

    # Do I wrap the cost function used as an argument to scipy.optimize.minimize
    # around an infidelity function which takes a propagator and its gradient
    # and calculates the infidelity and its gradient?
    # This is useful for writing different versions of the infidelity.
    ## For example, "spin echo infidelity cost",
    ## "dress control undress infidelity cost",
    ## "dress undress control infidelity cost"

    "costfunction/propagators/nsteps": int
    Number of time steps

    "costfunction/propagators/ncontrols": int
    Number of control fields

    "costfunction/propagators/nlandmarks": int
    Number of landmark points used for robustness

    "costfunction/propagators/ndim": int
    Dimension of the Hilbert space

    "costfunction/propagators/target": ndarray <complex> (ndim, ndim)
    target unitary transformation to be implemented

    "costfunction/propagators/weights": ndarray <float> (nlandmarks)
    Weights of each landmark point

    "costfunctions/propagators/tsteps": ndarray <float> (nsteps)
    Duration of time steps

    "costfunctions/propagators/control_variables_min":
        ndarray <real> (ncontrols)
    Minimum value of each control variable

    "costfunctions/propagators/control_variables_max":
            ndarray <real> (ncontrols)
    Maximum value of each control variable

    "costfunction/propagators/static_parameters":
        ndarray (nparameters, nlandmarks)
    Values of the static parameters for the Hamiltonian, that may vary across
    different landmark points. These are typically kept constant and may be
    subject to experimental uncertainty

    "costfunction/propagators/hamiltonian_function": function
    Python function which computes the Hamiltonian matrix using the static
    parameters and the control variables

    "costfunction/propagators/full_unitary_function": function
    Python function which computes the full unitary that depends on the unitary
    generated during the control

    Reads the following keys for the minimization process

    "minimize/input/gradient_tolerance": float
    Tolerance of the gradient for the minimization process

    "minimize/input/maxiter": int
    Maximum number of iterations for the minimization process

    "minimize/input/method": str
    Method used for minimization. The default is "BFGS"

    "minimize/input/initialization": str
    Choice of initial guess. "Random" for random, "Constant" for constant,
    "Sine" for sinusoidal, "Linear" for linear


    Returns
    -------
    control_variables_optimized: ndarray <real> (nsteps, ncontrols)
    Control variables found using optimal control

    infidelity_min:
    Minimum infidelity found using the optimization

    Side Affects
    ------------
    control_problem: dictionary

    Updates the following keys of the dictionary

    "minimize/output/niterations": int
    Number of iterations

    "minimize/output/infidelity_min": float
    Minimum infidelity

    "minimize/output/grad_infidelity_min":
        ndarray <float> (nsteps * ncontrols)
    Derivative of the minimum infidelity at the end of the minimization

    "minimize/output/control_variables_optimized":
        ndarray <float> (nsteps, ncontrols)
    Values of control variables at the end of the minimization
    """
    # Read the parameters of the control_problem
    nsteps = control_problem['costfunction/propagators/nsteps']
    ncontrols = control_problem['costfunction/propagators/ncontrols']
    nlandmarks = control_problem['costfunction/propagators/nlandmarks']
    ndim = control_problem['costfunction/propagators/ndim']

    tsteps = control_problem['costfunction/propagators/tsteps']

    initialization = control_problem['minimize/input/initialization']
    method = control_problem['minimize/input/method']
    gtol = control_problem['minimize/input/gradient_tolerance']
    matiter = control_problem['minimize/input/maxiter']

    control_variables_min = control_problem['control_variables_min']
    control_variables_max = control_problem['control_variables_max']

    # Selecting an initial guess
    if initialization == None:
        initialization = 'linear'
        
    control_variables_initial = initialize_guess(initialization, \
        nsteps, ncontrols, tsteps, control_variables_max, \
        control_variables_min)

    control_variable_initial_flattened = \
        np.reshape(control_variables_initial, (ncontrols * nsteps, )),

    # Using scipy.optimize.minimize to execute the minimization function
    result_minimize = scipy.optimize.minimize(\
                 fun=control_variable_initial_flattened, \
                 x0=control_variable_initial_reshaped
                 jac=True, method=method, \
                 options={'gtol': gtol, 'maxiter': maxiter,}, \
                 args=(control_problem,))

    # Reading the results of the optimization
    control_variables_optimized = \
        np.reshape(result_minimize.x, (ncontrols, nsteps))
    infidelity_min = result_minimize.fun
    grad_infidelity_min = result_minimize.jac
    status = result_minimize.status
    niterations = result_minimize.nit

    # Saving the results
    control_problem['minimize/output/control_variables_optimized'] \
        = control_variables_optimized

    control_problem['minimize/output/infidelity_min'] \
        = infidelity_min

    control_problem['minimize/output/grad_infidelity_min'] \
        = grad_infidelity_min

    control_problem['minimize/output/niterations'] = niterations

    control_problem['minimize/debug/result_minimize'] = result_minimize

    return control_variables, infidelity_min
