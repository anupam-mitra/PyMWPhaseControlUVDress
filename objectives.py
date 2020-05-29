#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  objectives.py
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
from __future__ import division

import copy
import itertools
import numpy as np
import fidelity
from timeevolution import propagator, propagator_with_gradient

def infidelity (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagainfidelitytor
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.get('PropagatorParameters')
    
    ket_initial = control_params.get('PureStateInitial')
    ket_target = control_params.get('PureStateTarget')
    u = propagator(phi, u_params)
    return fidelity.state_infidelity(u, ket_initial, ket_target)

def infidelity_gradient (phi, control_params):
    """
    Computes the gradient of infidelity for a control 
    task for given values of the control variables
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_gradient:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """
    u_params = control_params.get('PropagatorParameters')
    ket_initial = control_params.get('PureStateInitial')
    ket_target = control_params.get('PureStateTarget')
    u, u_gradient = propagator_with_gradient(phi, u_params)
    return fidelity.state_infidelity_gradient(u, u_gradient, ket_initial, ket_target)


def infidelity_unitary (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagainfidelitytor
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.get('PropagatorParameters')
    
    u_target = control_params.get('UnitaryTarget')
    Nstates = control_params.get('NStatesUnitary')
    
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['NStatesUnitary'] = Nstates
         
    u = propagator(phi, u_params)
    return fidelity.unitary_infidelity(u, u_target, Nstates)

def infidelity_unitary_gradient (phi, control_params):
    """
    Computes the gradient infidelity for a control task 
    for given values of the control variables
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagainfidelitytor
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.get('PropagatorParameters')
    
    u_target = control_params.get('UnitaryTarget')
    Nstates = control_params.get('NStatesUnitary')
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['NStatesUnitary'] = Nstates

    u, u_gradient = propagator_with_gradient(phi, u_params)
    return fidelity.unitary_infidelity_gradient(u, u_gradient, u_target, Nstates)

def _delta_r_pairs(control_params):
    h_inhomo = control_params['HamiltonianUncertainParameters']
    deltaR_values = h_inhomo['deltaDeltaRValues']
    return itertools.product(deltaR_values, deltaR_values), len(deltaR_values)**2

def _set_delta_r(control_params, deltaRa, deltaRb):
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    h_params = control_params['PropagatorParameters']['HamiltonianParameters']
    h_params['DeltaRa'] = DeltaR + deltaRa
    h_params['DeltaRb'] = DeltaR + deltaRb

def robust_infidelity (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    for inhomogeneities
    
    As of now it is very specific to Rydberg atoms
    
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_mean:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities
    """
  
    I_mean = 0
    delta_r_pairs, Nterms_average = _delta_r_pairs(control_params)

    for deltaRa, deltaRb in delta_r_pairs:
        _set_delta_r(control_params, deltaRa, deltaRb)
        I_mean += infidelity(phi, control_params)
    
    I_mean /= Nterms_average
    
    return I_mean

def robust_infidelity_gradient (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    for inhomogeneities
    
    As of now it is very specific to Rydberg atoms
    
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_gradient_mean:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """

    Nsteps = control_params['PropagatorParameters']['Nsteps']
    I_gradient_mean = np.zeros(Nsteps)
    delta_r_pairs, Nterms_average = _delta_r_pairs(control_params)

    for deltaRa, deltaRb in delta_r_pairs:
        _set_delta_r(control_params, deltaRa, deltaRb)
        I_gradient_mean += infidelity_gradient(phi, control_params)
        
    I_gradient_mean /= Nterms_average
    
    return I_gradient_mean

def robust_infidelity_unitary(phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    for inhomogeneities
    
    As of now it is very specific to Rydberg atoms
    
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_mean:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities
    """

    I_mean = 0
    delta_r_pairs, Nterms_average = _delta_r_pairs(control_params)

    for deltaRa, deltaRb in delta_r_pairs:
        _set_delta_r(control_params, deltaRa, deltaRb)
        I_mean += infidelity_unitary(phi, control_params)
    
    I_mean /= Nterms_average
    
    return I_mean

def robust_infidelity_unitary_gradient (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    for inhomogeneities
    
    As of now it is very specific to Rydberg atoms
    
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_gradient_mean:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """

    Nsteps = control_params['PropagatorParameters']['Nsteps']
    I_gradient_mean = np.zeros(Nsteps)
    delta_r_pairs, Nterms_average = _delta_r_pairs(control_params)

    for deltaRa, deltaRb in delta_r_pairs:
        _set_delta_r(control_params, deltaRa, deltaRb)
        I_gradient_mean += infidelity_unitary_gradient(phi, control_params)
        
    I_gradient_mean /= Nterms_average
    
    return I_gradient_mean

def robust_adiabatic_infidelity_unitary(phi, control_params):
    """
    Computes the infidelity for a control task
    for given values of the control variables
    for inhomogeneities and the gradient of the
    infidelity for the same task
    
    As of now it is very specific to Rydberg atoms
    
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    Dictionary representing parameters of the control problem
    with the following keys
    
    Returns
    -------
    I_average:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities
    
    I_average_gradient:
    Gradient of infidelity = 1 - fidelity with respect to the
    control variables evaluated at given values of the control
    variables
    """

    hamiltonian_landmarks_list = control_params['HamiltonianLandmarks']
    Nlandmarks = len(hamiltonian_landmarks_list)

    Nsteps = control_params['PropagatorParameters']['Nsteps']

    u_params = control_params.get('PropagatorParameters')

    u_target = control_params.get('UnitaryTarget')
    Nstates = control_params.get('NStatesUnitary')

    # Check for the matrix rank of the desired unitary to use for calculating
    # infidelity for a partial isometry
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['NStatesUnitary'] = Nstates

    F_average = 0
    F_average_gradient = np.zeros(Nsteps)

    Infidelities = np.zeros((Nlandmarks,))

    unitary_dressing_landmarkwise = []
    unitary_undressing_landmarkwise = []
    unitary_mwcontrol_landmarkwise = []
    unitary_mwcontrol_gradient_landmarkwise = []

    if 'LandmarkWeights' in control_params:
        landmark_weights = control_params.get('LandmarkWeights')
    else:
        landmark_weights = np.ones((Nlandmarks, )) / Nlandmarks

    for l in range(Nlandmarks):
        hamiltonian_landmark_current = hamiltonian_landmarks_list[l]

        DeltaRa = hamiltonian_landmark_current.get('DeltaRa')
        DeltaRb = hamiltonian_landmark_current.get('DeltaRb')

        u_params_current = copy.deepcopy(u_params)

        u_params_current['HamiltonianParameters']['DeltaRa'] \
        = DeltaRa
        u_params_current['HamiltonianParameters']['DeltaRb'] \
        = DeltaRb

        u, u_gradient = propagator_with_gradient(phi, u_params_current)

        u_dress = control_params['UnitaryDressingLandmarks'][l]
        u_undress = control_params['UnitaryUnDressingLandmarks'][l]

        u_protocol = np.dot(u_undress, np.dot(u, u_dress))

        F = fidelity.unitary_fidelity(u_protocol, u_target, Nstates)
        F_average += landmark_weights[l] * F

        for n in range(Nsteps):
            u_protocol_gradient_step = np.dot(u_undress, np.dot(u_gradient[n], u_dress))
            F_average_gradient[n] -= landmark_weights[l] * fidelity.unitary_infidelity_gradient(
                u_protocol, [u_protocol_gradient_step], u_target, Nstates)[0]

    I_average = 1 - F_average
    I_average_gradient = - F_average_gradient

    return I_average, I_average_gradient

# I need to fix the robust_adiabatic_infidelity_unitary function implementation details
