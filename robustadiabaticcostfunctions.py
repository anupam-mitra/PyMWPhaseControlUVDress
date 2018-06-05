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

import itertools
import numpy as np
import costfunctions
import propagators

def infidelity_unitary(phi, control_params):
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

    h_inhomo = control_params['HamiltonianUncertainParameters']
    
    deltaR_values = h_inhomo['deltaDeltaRValues']
    
    Nterms_average = len(deltaR_values)**2
    Nsteps = control_params['PropagatorParameters']['Nsteps']
    
    DeltaR = control_params['HamiltonianBaseParameters']['DeltaR']
    
    u_params = control_params.get('PropagatorParameters')
    
    u_target = control_params.get('UnitaryTarget')
    Nstates = control_params.get('NStatesUnitary')
    
    u_dress_dict = control_params.get('UnitaryDressing')
    u_undress_dict = control_params.get('UnitaryUndressing')
    
    u_target_dagger = u_target.conjugate().transpose()
    
    # Check for the matrix rank of the desired unitary to use for calculating
    # infidelity for a partial isometry
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['Nstates'] = Nstates
        
    F_average = 0
    F_average_gradient = np.zeros(Nsteps)
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        control_params['PropagatorParameters']['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
      
        u, u_gradient  = propagators.propagator(phi, u_params)
        udagger = u.transpose().conjugate()
    
        u_dress = u_dress_dict.get((deltaRa, deltaRb))
        u_undress = u_dress_dict.get((deltaRa, deltaRb))
    
        u_protocol = np.dot(u_dress, np.dot(u, u_undress))
        u_protocol_dagger = np.conjugate(np.transpose(u_protocol))
        
        F = np.abs(np.trace(np.dot(u_protocol, u_target_dagger)))**2 / Nstates**2
        F_average += F 
        
        for n in range(Nsteps):
            u_protocol_gradient_step = np.dot(u_undress, np.dot(u_gradient[n], u_dress))
        
            F_average_gradient[n] += 2*np.real(\
              np.trace(np.dot(u_target_dagger, u_protocol_gradient_step))
              * np.trace(np.dot(u_target, u_protocol_dagger))) / Nstates**2


    F_average = F_average/Nterms_average
    I_average = 1 - F_average

    F_average_gradient /= Nterms_average        
    I_average_gradient = - F_average_gradient


    return I_average, I_average_gradient

