#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propagators.py
#
#  Copyright 2018 Anupam Mitra <anupam@unm.edu>
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

import utilities

from propagators import propagator_step_derivative

from utilities import dagger

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
