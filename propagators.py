#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propagators.py
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

import itertools
import numpy as np
import scipy.linalg

from scipy.linalg import expm as expm

def propagator_step_derivative(h_step, h_step_deriv, Tstep):
    """
    Computes the derivative of a step of the piecewise constant
    propagator with respect to particular control variable, given
    - the hamiltonian matrix during the control step, evaluated at
     appropriate values of the control variables
    - the derivative of the hamiltonian matrix with respect to
    the appropriate control variable, evaluated at
     appropriate values of the control variables

    Parameters
    ----------
    h_step:
    Hamiltonian matrix during the step for which the gradient of
    the propagator is computed

    h_step_deriv:
    Derivative of the hamiltonian matrix with respect to
    the control variable in question during the step for which the gradient of
    the propagator is computed

    Tstep:
    Temporal duration of the time step

    Returns
    -------
    u_step_derivative:
    Matrix which is the derivative of the propagator with respect
    to a particular control variable
    """

    eigenvalues, eigenvectors = scipy.linalg.eigh(h_step)
    u_step_derivative = np.zeros(eigenvectors.shape, dtype=complex)

    for (a, ea), (b, eb) in itertools.product(enumerate(eigenvalues), enumerate(eigenvalues)):
        outerprod = np.outer(eigenvectors[:, a], eigenvectors[:, b].conjugate())
        h_step_grad_matrixel = \
            np.dot(eigenvectors[:, a].conjugate(), np.dot(h_step_deriv, eigenvectors[:, b]))
        if ea == eb:
            u_step_derivative += -1j * np.exp(-1j*Tstep*ea) * outerprod * h_step_grad_matrixel
        else:
            u_step_derivative += (np.exp(-1j*Tstep*ea) - np.exp(-1j*Tstep*eb))/(ea - eb) \
                                        * outerprod * h_step_grad_matrixel

    return u_step_derivative

def _propagator_steps(phi, propagator_params, gradient=False):
    Nsteps = propagator_params['Nsteps']
    Tstep = propagator_params['Tstep']
    hamiltonian_func = propagator_params['HamiltonianMatrix']
    h_params = propagator_params['HamiltonianParameters']

    h_steps = [hamiltonian_func(phi[n], h_params) for n in range(Nsteps)]
    u_steps = [expm(-1j*Tstep*h) for h in h_steps]

    if not gradient:
        return h_steps, u_steps

    hamiltonian_grad_func = propagator_params['HamiltonianMatrixGradient']
    h_grad_steps = [hamiltonian_grad_func(phi[n], h_params) for n in range(Nsteps)]
    return h_steps, u_steps, h_grad_steps

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

        u_step_derivative = propagator_step_derivative(
                                        h_steps[n], h_grad_steps[n], Tstep)
        u_gradient[n] = np.dot(u_step_derivative, u_gradient[n])

        for m in range(n+1, len(h_steps)):
            u_gradient[n] = np.dot(u_steps[m], u_gradient[n])

    return u_gradient

def propagator_gradient (phi, propagator_params):
    """
    Computes the gradient of the propagator with respect to
    the control variables, at given values of the control
    variables

    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator

    propagator_params:
    Dictionary representing parameters of the propagator with
    the following keys

    Returns
    -------
    u_gradient:
    Gradient of the propagator with respect to the control
    variables, evaluated at given values of the control
    variables
    """

    Tstep = propagator_params['Tstep']
    h_steps, u_steps, h_grad_steps = _propagator_steps(phi, propagator_params, True)

    return _gradient_from_steps(h_steps, u_steps, h_grad_steps, Tstep)


def propagator (phi, propagator_params):
    """
    Computes the propagator at given values of the control variables

    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator

    propagator_params:
    Dictionary representing parameters of the propagator with
    the following keys

    Returns
    -------
    u:
    Propagator evaluated at given values of the control
    variables
    """
    h_steps, u_steps = _propagator_steps(phi, propagator_params)

    return _propagator_from_steps(u_steps)

def propagator_with_gradient (phi, propagator_params):
    """
    Computes the propagator and its gradient at given control variables.
    """
    Tstep = propagator_params['Tstep']
    h_steps, u_steps, h_grad_steps = _propagator_steps(phi, propagator_params, True)

    return (
        _propagator_from_steps(u_steps),
        _gradient_from_steps(h_steps, u_steps, h_grad_steps, Tstep),
    )
