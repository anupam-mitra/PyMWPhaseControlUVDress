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

import copy
import itertools
import numpy as np

from core import fidelity
from core.evolution import propagator, propagator_with_gradient
from core.params import ControlProblem

def infidelity (phi, control_params):
    """
    Computes the infidelity for a control task 
    for given values of the control variables
    Parameters
    ----------
    phi:
    Values of the control variables at which to evaluate the
    gradient of the propagator
    
    control_params:
    ControlProblem dataclass representing parameters of the control problem
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.propagator_params
    
    ket_initial = control_params.pure_state_initial
    ket_target = control_params.pure_state_target
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
    ControlProblem dataclass representing parameters of the control problem
    
    Returns
    -------
    I_gradient:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """
    u_params = control_params.propagator_params
    ket_initial = control_params.pure_state_initial
    ket_target = control_params.pure_state_target
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
    ControlProblem dataclass representing parameters of the control problem
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.propagator_params
    
    u_target = control_params.unitary_target
    Nstates = control_params.n_states_unitary
    
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params.n_states_unitary = Nstates
          
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
    ControlProblem dataclass representing parameters of the control problem
    
    Returns
    -------
    I:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables
    """
        
    u_params = control_params.propagator_params
    
    u_target = control_params.unitary_target
    Nstates = control_params.n_states_unitary
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params.n_states_unitary = Nstates

    u, u_gradient = propagator_with_gradient(phi, u_params)
    return fidelity.unitary_infidelity_gradient(u, u_gradient, u_target, Nstates)


def _delta_r_pairs(control_params):
    h_inhomo = control_params.hamiltonian_uncertain_params
    deltaR_values = h_inhomo['deltaDeltaRValues']
    return itertools.product(deltaR_values, deltaR_values), len(deltaR_values)**2

def _set_delta_r(control_params, deltaRa, deltaRb):
    DeltaR = control_params.hamiltonian_base_params['DeltaR']
    h_params = control_params.propagator_params.hamiltonian_params
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
    ControlProblem dataclass or dictionary representing parameters of the control problem
    
    Returns
    -------
    I_mean:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities
    """
  
    if isinstance(control_params, dict):
        control_params = ControlProblem.from_dict(control_params)
        
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
    ControlProblem dataclass or dictionary representing parameters of the control problem
    
    Returns
    -------
    I_gradient_mean:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """
    if isinstance(control_params, dict):
        control_params = ControlProblem.from_dict(control_params)

    Nsteps = control_params.propagator_params.Nsteps
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
    ControlProblem dataclass or dictionary representing parameters of the control problem
    
    Returns
    -------
    I_mean:
    Infidelity = 1 - fidelity evaluated at given values of the control
    variables, averaged over all inhomogeneities
    """
    if isinstance(control_params, dict):
        control_params = ControlProblem.from_dict(control_params)

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
    ControlProblem dataclass or dictionary representing parameters of the control problem
    
    Returns
    -------
    I_gradient_mean:
    Gradient of infidelity = 1 - fidelity with respect to the 
    control variables evaluated at given values of the control
    variables
   
    """
    if isinstance(control_params, dict):
        control_params = ControlProblem.from_dict(control_params)

    Nsteps = control_params.propagator_params.Nsteps
    I_gradient_mean = np.zeros(Nsteps)
    delta_r_pairs, Nterms_average = _delta_r_pairs(control_params)

    for deltaRa, deltaRb in delta_r_pairs:
        _set_delta_r(control_params, deltaRa, deltaRb)
        I_gradient_mean += infidelity_unitary_gradient(phi, control_params)
        
    I_gradient_mean /= Nterms_average
    
    return I_gradient_mean
