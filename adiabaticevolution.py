#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  adibaticevolution.py
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

import numpy as np
import scipy

from numpy import exp, sin, cos, tan, arctan2
from scipy.linalg import expm

import rydbergatoms


"""
We compute adiabatic evolution for different cases. We first consider a variant
of rapid adiabatic passage, where keeping the Rabi frequency fixed, we sweep
to near resonance


"""

def adiabaticrydbergdressing_propagator_detuningsweep(adiabatic_params):
    """
    Computes the propagator for an adiabatic evolution where the 
    Hamiltonian is adiabatically changed from one Hamiltonian to another.
    
    This is a variant
	of rapid adiabatic passage, where keeping the Rabi frequency fixed, we sweep
	to near resonance
    
    Parameters
    ----------
            
    adiabatic_params:
        Parameters to use for adiabatic evolution
        

    Returns
    -------
    
    propagator_cumulative_dress:
        Propagator for the adiabatic dressing
        
    propagator_cumulative_undress:
        Propagator for the adiabatic undressing
    """
    
    Tstep = adiabatic_params.get('Tstep')
    Tsweepfactor = adiabatic_params.get('TSweepFactor')
    tinitial = adiabatic_params.get('tinitial')
    tfinal = adiabatic_params.get('tfinal')
        
    t_dress_initial = tinitial
    t_dress_final = 0
    t_undress_initial = 0
    t_undress_final = tfinal
    
    Ndim = adiabatic_params.get('DimensionHilbertSpace')
    
    hamiltonian_func = adiabatic_params.get('HamiltonianMatrix')
    hamiltonian_grad_func = adiabatic_params.get('HamiltonianMatrixGradient')
    
    h_params = adiabatic_params.get('HamiltonianParameters')

    t_dress = np.arange(t_dress_initial, t_dress_final + Tstep, Tstep)
    t_undress = np.arange(t_undress_initial, t_undress_final + Tstep, Tstep)
    
    # Linear Theta Ramp
    small_angle = adiabatic_params.get('SmallAngle')
    if small_angle == None:
        small_angle = 1/128
        
    DeltaRa = h_params.get('DeltaRa')
    DeltaRb = h_params.get('DeltaRb')
    OmegaRa = h_params.get('OmegaRa')
    OmegaRb = h_params.get('OmegaRb')
    
    thetaRa = arctan2(OmegaRa, -DeltaRa)
    thetaRb = arctan2(OmegaRb, -DeltaRb)
    
    thetaRa_dress = (small_angle + \
        (t_dress - t_dress_initial) / (t_dress_final - t_dress_initial) \
            * (1 - 2 * small_angle)) * thetaRa
    
    thetaRa_undress = np.flip(thetaRa_dress, axis=0)
    
    thetaRb_dress = (small_angle + \
        (t_undress - t_dress_initial) / (t_dress_final - t_dress_initial) \
            * (1 - 2 * small_angle)) * thetaRb
    
    thetaRb_undress = np.flip(thetaRb_dress, axis=0)

    OmegaRa_dress = DeltaRa / tan(thetaRa_dress)
    OmegaRb_dress = DeltaRb / tan(thetaRb_dress)

    OmegaRa_undress = DeltaRa / tan(thetaRa_undress)
    OmegaRb_undress = DeltaRb / tan(thetaRb_undress)

    Nsteps = thetaRa_undress.shape[0]
    
    hamiltonians_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_dress = np.identity(Ndim)

    hamiltonians_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_undress = np.identity(Ndim)
    
    for n in range(Nsteps):
        h_params_dress = {\
            'OmegaRa' : OmegaRa_dress[n], \
            'OmegaRb' : OmegaRb_dress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa, \
            'DeltaRb' : DeltaRb, \
            'DeltaMWa' : 0.01, \
            'DeltaMWb' : 0.01, \
        }
        
        h_params_undress = {\
            'OmegaRa' : OmegaRa_undress[n], \
            'OmegaRb' : OmegaRb_undress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa, \
            'DeltaRb' : DeltaRb, \
            'DeltaMWa' : 0.01, \
            'DeltaMWb' : 0.01, \
        }

        
        hamiltonians_dress[:, :, n] = \
            rydbergatoms.hamiltonian_PerfectBlockade(0, h_params_dress)
        propagators_dress[:, :, n] = expm(-1j * Tstep * hamiltonians_dress[:, :, n])

        propagator_cumulative_dress = np.dot(propagators_dress[:, :, n], propagator_cumulative_dress)
        
        hamiltonians_undress[:, :, n] = \
            rydbergatoms.hamiltonian_PerfectBlockade(0, h_params_undress)
        propagators_undress[:, :, n] = expm(-1j * Tstep * hamiltonians_undress[:, :, n])

        propagator_cumulative_undress = np.dot(propagators_undress[:, :, n], propagator_cumulative_undress)


    return propagator_cumulative_dress, propagator_cumulative_undress
    
