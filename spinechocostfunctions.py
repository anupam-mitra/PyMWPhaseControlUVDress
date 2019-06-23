#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  robustadiabaticcostfunctions.py
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

from numpy import sqrt

import copy
import itertools
import numpy as np
import costfunctions
import propagators
import rydbergatoms

def isinggate (axis):
    """
    Computes the unitary transformation for the
    ising gate along the direction specified

    Parameters
    ----------
    axis:
    Axis about which to perform rotations

    Returns
    -------
    u_ising:
    Unitary transformation corresponding to the Ising
    gate along the specified axis
    """

    Ndimensions = np.shape(rydbergatoms.ket_00)[0]

    if axis == 'y':
        sigmay_sigmay = \
            - np.outer(rydbergatoms.ket_00, rydbergatoms.bra_11) \
            - np.outer(rydbergatoms.ket_11, rydbergatoms.bra_00) \
            + np.outer(rydbergatoms.ket_01, rydbergatoms.bra_10) \
            + np.outer(rydbergatoms.ket_10, rydbergatoms.bra_01) \

        u_ising = 1/sqrt(2) * (np.eye(Ndimensions) - 1j * sigmay_sigmay)

    else:
        u_ising = np.eye(Ndimensions)


    u_ising = np.dot(rydbergatoms.projector_logical, u_ising)

    return u_ising

def infidelity_robust (phi, control_params):
    """
    Computes the infidelity for a robust control task

    Parameters
    ----------
    phi:
    Values of the control variables at which to
    evaluate the gradient of the propagator

    control_params:
    Dictionary representing parameters of the
    control problem

    Returns
    -------
    infidelity_mean:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities

    infidelity_gradient_mean:
    Gradient of infidelity = 1 - fidelity with respect to the
    control variables evaluated at given values of the control
    variables

    """

    hamiltonian_landmarks_list = control_params['HamiltonianLandmarks']
    Nlandmarks = len(hamiltonian_landmarks_list)

    u_params = control_params['PropagatorParameters']
    Nsteps = control_params['PropagatorParameters']['Nsteps']

    if 'LandmarkWeights' in control_params:
        landmark_weights = control_params.get('LandmarkWeights')
    else:
        landmark_weights = np.ones((Nlandmarks, )) / Nlandmarks

    infidelity_mean = 0

    infidelity_gradient_mean = np.zeros((Nsteps,))

    for l in range(Nlandmarks):
        hamiltonian_landmark_current = hamiltonian_landmarks_list[l]

        DeltaRa = hamiltonian_landmark_current.get('DeltaRa')
        DeltaRb = hamiltonian_landmark_current.get('DeltaRb')

        control_params_current = copy.deepcopy(control_params)

        u_params_current = copy.deepcopy(u_params)

        control_params_current['PropagatorParameters'] = \
            u_params_current

        u_params_current['HamiltonianParameters']['DeltaRa'] \
            = DeltaRa
        u_params_current['HamiltonianParameters']['DeltaRb'] \
            = DeltaRb

        # Read dressing and undressing unitaries
        u_dress = control_params['UnitaryDressingLandmarks'][l]
        u_undress = control_params['UnitaryUnDressingLandmarks'][l]

        # Add dressing and undressing unitaries to the control
        # problem dictionary
        control_params_current['UnitaryDressing'] = u_dress
        control_params_current['UnitaryUndressing'] = u_undress

        infidelity_current, infidelity_gradient_current = \
            infidelity(phi, control_params_current)

        infidelity_mean += infidelity_current * landmark_weights[l]
        infidelity_gradient_mean += infidelity_gradient_current \
                                        * landmark_weights[l]

    return infidelity_mean, infidelity_gradient_mean

def infidelity (phi, control_params):
    """
    Computes the infidelity for a control task
    for given values of control variables and the
    gradient of the infidelity of the same task

    Parameters
    ----------
    phi:
    Values of the control variables at which to
    evaluate the gradient of the propagator

    control_params:
    Dictionary representing parameters of the
    control problem

    Returns
    -------
    infidelity:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities

    infidelity_gradient:
    Gradient of infidelity = 1 - fidelity with respect to the
    control variables evaluated at given values of the control
    variables

    """

    hamiltonian_landmarks_list = control_params['HamiltonianLandmarks']
    Nlandmarks = len(hamiltonian_landmarks_list)

    Nterms_average = Nlandmarks
    Nsteps = control_params['PropagatorParameters']['Nsteps']

    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']

    u_params = control_params.get('PropagatorParameters')

    u_target = control_params.get('UnitaryTarget')
    if u_target == None:
        u_target = isinggate('y')
        control_params['UnitaryTarget'] = u_target

    Nstates = control_params.get('NStatesUnitary')

    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)

    u_dress = control_params.get('UnitaryDressing')
    u_undress = control_params.get('UnitaryUndressing')

    u_mw_x_halfpi = control_params.get('UnitaryMWXHalfPi')
    u_mw_x_pi = control_params.get('UnitaryMWXPi')

    u_target_dagger = u_target.conjugate().transpose()

    u, u_gradient = propagators.propagator(phi, u_params)

    u_protocol = np.dot(u_mw_x_halfpi, \
                    np.dot(u_undress, \
                        np.dot(u, \
                            np.dot(u_dress, \
                                np.dot(u_mw_x_pi, \
                                    np.dot(u_undress, \
                                        np.dot(u, \
                                            np.dot(u_dress, \
                                                u_mw_x_halfpi,))))))))

    goal = np.trace(np.dot(u_protocol, u_target_dagger))
    infidelity = 1 - 1/(Nstates**2) * (np.abs(goal))**2

    goal_gradient = np.zeros((Nsteps,), dtype=complex)
    infidelity_gradient = np.zeros((Nsteps,))

    for n in range(Nsteps):
        u_protocol_gradient_step = \
            np.dot(u_mw_x_halfpi, \
                    np.dot(u_undress, \
                        np.dot(u_gradient[n], \
                            np.dot(u_dress, \
                                np.dot(u_mw_x_pi, \
                                    np.dot(u_undress, \
                                        np.dot(u, \
                                            np.dot(u_dress, \
                                                u_mw_x_halfpi,)))))))) \
          + np.dot(u_mw_x_halfpi, \
                    np.dot(u_undress, \
                        np.dot(u, \
                            np.dot(u_dress, \
                                np.dot(u_mw_x_pi, \
                                    np.dot(u_undress, \
                                        np.dot(u_gradient[n], \
                                            np.dot(u_dress, \
                                                u_mw_x_halfpi,))))))))

        goal_gradient[n] = np.trace(np.dot(\
                            u_protocol_gradient_step, u_target_dagger))

        infidelity_gradient[n] = - 2/(Nstates**2) * \
             np.real(goal_gradient[n] * np.conjugate(goal))


    return infidelity, infidelity_gradient
