#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  timeevolution.py
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
#

import itertools
import numpy as np
import scipy.linalg
from scipy.linalg import expm as expm

def dagger(u):
    return np.transpose(np.conjugate(u))

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

def calc_time_evolve_op(hamiltonians, grad_hamiltonians, tsteps, gradient=True):
    """
    Calculates the time evolution operator and its gradient with respect to
    control variables for a piecewise constant time depdendent Hamiltonian.

    Parameters
    ----------
    hamiltonians: ndarray (ndim, ndim, nlandmarks, nsteps)
    Hamiltonian matrices at each time step. The zeroth and first indices
    determine the matrix element. The second index determines the landmark
    point. The third index determines the time step.

    grad_hamiltonians: ndarray (ndim, ndim, nlandmarks, nsteps, ncontrols)
    Gradient of the Hamiltonian matrix with respect to control variables.
    The zeroth and the first indices determine the matrix elements. The
    second index determines the landmark point. The third and the fourth
    indices determine the component of the
    gradient.

    tsteps: ndarray (nsteps)
    Time duration of Hamiltonian steps (or pieces). This is represented as an
    array to facilitate different durations during control.

    ndim: int
    Dimension of the Hilbert space

    Returns
    -------
    propagator_cumulative: ndarray (ndim, ndim, nlandmarks)
    Time evolution operator. The zeroth and the first index determine the matrix
    elements. The second index determines the landmark point.

    grad_propagator_cumulative: ndarray (ndim, ndim, nlandmarks, nsteps, ncontrols)
    The gradient of the time evolution operator. The zeroth and the first indices
    determine the matrix elements. The second index determines the landmark
    point. The third and the fourth indices determine the component of the
    gradient.
    """

    hamiltonians_shape_tuple = np.shape(hamiltonians)
    grad_hamiltonians_shape_tuple = np.shape(grad_hamiltonians)

    assert hamiltonians_shape_tuple[0] == hamiltonians_shape_tuple[1] == \
        grad_hamiltonians_shape_tuple[0] == grad_hamiltonians_shape_tuple[1] \

    assert hamiltonians_shape_tuple[2] == grad_hamiltonians_shape_tuple[2]

    assert hamiltonians_shape_tuple[3] == grad_hamiltonians_shape_tuple[3]

    ndim = hamiltonians_shape_tuple[0]
    nlandmarks = hamiltonians_shape_tuple[2]
    nsteps = hamiltonians_shape_tuple[3]

    ncontrols = grad_hamiltonians_shape_tuple[4]

    propagators \
        = np.zeros((ndim, ndim, nlandmarks, nsteps+1, nsteps+1), dtype=complex)
    grad_propagators \
        = np.zeros((ndim, ndim, nlandmarks, nsteps+1, nsteps+1, nsteps, ncontrols), dtype=complex)

    for l in range(nlandmarks):

        propagators[:, :, l, 0, 0] = np.eye(ndim, ndim)

        for s_begin in range(nsteps):
            propagators[:, :, l, s_begin+1, s_begin+1] = np.eye(ndim, ndim)
            propagators[:, :, l, s_begin+1, s_begin] \
                = expm(-1j * tsteps[s_begin] * hamiltonians[:, :, l, s_begin])
            propagators[:, :, l, s_begin, s_begin+1] \
                = dagger(propagators[:, :, l, s_begin+1, s_begin])

        for s_begin in range(nsteps):
            for s_end in range(s_begin+1, nsteps+1):
                propagators[:, :, l, s_end, s_begin] \
                    = np.dot(propagators[:, :, l, s_end, s_end-1],\
                            propagators[:, :, l, s_end-1, s_begin])
                propagators[:, :, l, s_begin, s_end] \
                    = dagger(propagators[:, :, l, s_end, s_begin])

        for k in range(ncontrols):
            for s_begin in range(nsteps):
                grad_propagators[:, :, l, s_begin+1, s_begin, s_begin, k] \
                    = propagator_step_derivative(\
                       hamiltonians[:, :, l, s_begin], \
                       grad_hamiltonians[:, :, l, s_begin, k], \
                       tsteps[s_begin])
                grad_propagators[:, :, l, s_begin, s_begin+1, s_begin, k] \
                    = dagger(grad_propagators[:, :, l, s_begin+1, s_begin, s_begin, k])


            for s_begin in range(nsteps):
                for s_end in range(s_begin+1, nsteps+1):
                    for s in range(s_begin, s_end):
                        grad_propagators[:, :, l, s_end, s_begin, s, k] \
                        = np.dot(\
                            propagators[:, :, l, s_end, s+1], \
                            np.dot(\
                                    grad_propagators[:, :, l, s+1, s, s, k],\
                                    propagators[:, :, l, s, s_begin]))

                        grad_propagators[:, :, l, s_begin, s_end, s, k] \
                            = dagger(grad_propagators[:, :, l, s_end, s_begin, s, k])

    if gradient:
        return propagators, grad_propagators
    else:
        return propagators
