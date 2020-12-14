#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  robustcostfunctions.py
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
import costfunctions

def infidelity (phi, control_params):
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
 
    h_inhomo = control_params['HamiltonianUncertainParameters']
    
    #deltaOmega_values = h_inhomo['deltaOmegaValues']
    deltaR_values = h_inhomo['deltaDeltaRValues']
    
    I_mean = 0
    #Nterms_average = len(deltaOmega_values) * len(deltaR_values)
    Nterms_average = len(deltaR_values)**2
    
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
        
        I_mean += costfunctions.infidelity(phi, control_params)
    
    I_mean /= Nterms_average
    
    return I_mean
    
    
def infidelity_gradient (phi, control_params):
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

    h_inhomo = control_params['HamiltonianUncertainParameters']
    
    #deltaOmega_values = h_inhomo['deltaOmegaValues']
    deltaR_values = h_inhomo['deltaDeltaRValues']
    
    #Nterms_average = len(deltaOmega_values) * len(deltaR_values)
    Nterms_average = len(deltaR_values)**2
    Nsteps = control_params['PropagatorParameters']['Nsteps']
    
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    
    I_gradient_mean = np.zeros(Nsteps)
    
    
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
        
        I_gradient_mean = costfunctions.infidelity_gradient(phi, control_params)
        
    I_gradient_mean /= Nterms_average
    
    return I_gradient_mean

def infidelity_unitary(phi, control_params):
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

    h_inhomo = control_params['HamiltonianUncertainParameters']
    
    #deltaOmega_values = h_inhomo['deltaOmegaValues']
    deltaR_values = h_inhomo['deltaDeltaRValues']
    
    I_mean = 0
    #Nterms_average = len(deltaOmega_values) * len(deltaR_values)
    Nterms_average = len(deltaR_values)**2
    
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
        
        I_mean += costfunctions.infidelity_unitary(phi, control_params)
    
    I_mean /= Nterms_average
    
    return I_mean

def infidelity_unitary_gradient (phi, control_params):
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

    h_inhomo = control_params['HamiltonianUncertainParameters']
    
    #deltaOmega_values = h_inhomo['deltaOmegaValues']
    deltaR_values = h_inhomo['deltaDeltaRValues']
    
    #Nterms_average = len(deltaOmega_values) * len(deltaR_values)
    Nterms_average = len(deltaR_values)**2
    Nsteps = control_params['PropagatorParameters']['Nsteps']
    
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    
    I_gradient_mean = np.zeros(Nsteps)
    
    
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
        
        I_gradient_mean = costfunctions.infidelity_unitary_gradient(phi, control_params)
        
    I_gradient_mean /= Nterms_average
    
    return I_gradient_mean
