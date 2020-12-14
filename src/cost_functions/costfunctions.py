#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  costfunctions.py
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

from propagators import propagator
from propagators import propagator_gradient

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
    
    bra_target = ket_target.transpose().conjugate()
    
    u = propagator(phi, u_params)
    
    F = np.abs(np.dot(bra_target, np.dot(u, ket_initial))[0, 0])**2
    I = 1 - F

    return I

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
    Nsteps = u_params.get('Nsteps')
    
    ket_initial = control_params.get('PureStateInitial')
    ket_target = control_params.get('PureStateTarget')
    
    bra_target = ket_target.transpose().conjugate()
    
    u = propagator(phi, u_params)
    u_gradient = propagator_gradient(phi, u_params)
    
    F_gradient = np.zeros(Nsteps)
    
    for n in range(Nsteps):
        F_gradient[n] = 2*np.real(\
            np.conjugate(np.dot(bra_target, np.dot(u, ket_initial)))[0, 0] \
            * np.dot(bra_target, np.dot(u_gradient[n], ket_initial))[0, 0])
    
    I_gradient = -F_gradient
    return I_gradient


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
    
    # Check for the matrix rank of the desired unitary to use for calculating
    # infidelity for a partial isometry
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['Nstates'] = Nstates
        
    u = propagator(phi, u_params)
    udagger = u.transpose().conjugate()
    
    F = np.abs(np.trace(np.dot(udagger, u_target)))**2 / Nstates**2
    I = 1 - F

    return I

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
    Nsteps = control_params.get('PropagatorParameters').get('Nsteps')
    
    # Check for the matrix rank of the desired unitary to use for calculating
    # infidelity for a partial isometry
    if Nstates == None:
        Nstates = np.linalg.matrix_rank(u_target)
        control_params['Nstates'] = Nstates
        
    u_targetdagger = u_target.conjugate().transpose()

    u = propagator(phi, u_params)
    udagger = u.transpose().conjugate()
    
    u = propagator(phi, u_params)
    u_gradient = propagator_gradient(phi, u_params)
    

    F_gradient = np.zeros(Nsteps)
    
    #TODO
    for n in range(Nsteps):
        F_gradient[n] = 2*np.real(\
                  np.trace(np.dot(u_targetdagger, u_gradient[n]))
                  * np.trace(np.dot(u_target, udagger))) / Nstates**2
    
    I_gradient = -F_gradient
    return I_gradient

    
