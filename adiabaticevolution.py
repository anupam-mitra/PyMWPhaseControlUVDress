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
    

def adiabatic_evolution_propagators (adiabatic_parameters):
    """
    Computes the propagator for an adiabatic evolution where the 
    Hamiltonian is adiabatically changed from one Hamiltonian to another.
    
    The detuning is swept linearly towards resonance and the the Rabi
    frequency is swept from 0 to a maximum value as a Gaussian during
    dressing.

    The detuning is swept linearly away from resonance and the the Rabi
    frequency is swept from a maximum value to 0 as a Gaussian during
    undressing.

    # A better way to perform compute this propagator is as follows
    # Use scipy.integrate.ode to inegrate the time dependent Schrodinger
    # equation for the dressing to obtain the dressing unitary u_dress
    # and for the undressing unitary to obtain the undressing unitary
    # u_undress

    Parameters
    ----------
            
    adiabatic_parameters:
        Parameters to use for adiabatic evolution
        

    Returns
    -------
    
    propagator_cumulative_dress:
        Propagator for the adiabatic dressing
        
    propagator_cumulative_undress:
        Propagator for the adiabatic undressing
    """

    t_gaussian_width = adiabatic_parameters['t_gaussian_width']
    DeltaMW = adiabatic_parameters['DeltaMW']

    DeltaRa_min = adiabatic_parameters['DeltaR_min']
    DeltaRa_max = adiabatic_parameters['DeltaR_max']

    DeltaRb_min = adiabatic_parameters['DeltaR_min']
    DeltaRb_max = adiabatic_parameters['DeltaR_max']

    OmegaR_min = adiabatic_parameters['OmegaR_min']
    OmegaR_max = adiabatic_parameters['OmegaR_max']

    Ndim = adiabatic_parameters['DimensionHilbertSpace']
    
    Nsteps = adiabatic_parameters.get('Nsteps')
    if Nsteps == None:
        Nsteps = 1024

    hamiltonian_func = adiabatic_parameters.get('HamiltonianMatrix')

    t_dress_begin = - 4 * t_gaussian_width
    t_dress_end = 0
    t_undress_begin = 0
    t_undress_end = 4 * t_gaussian_width

    t_dress = np.linspace(t_dress_begin, t_dress_end, num=Nsteps)
    t_undress = np.linspace(t_undress_begin, t_undress_end, num=Nsteps)

    Tstep = (t_dress_end - t_dress_begin)/Nsteps
    
    OmegaRa_t_dress = OmegaR_min + (OmegaR_max - OmegaR_min) \
                * exp(-(t_dress - t_dress_end) / 2 / t_gaussian_width**2)

    OmegaRb_t_dress = OmegaR_min + (OmegaR_max - OmegaR_min) \
                * exp(-(t_dress - t_dress_end) / 2 / t_gaussian_width**2)

    DeltaRa_t_dress = DeltaRa_max + \
            (DeltaRa_min - DeltaRa_max) / (t_dress_end - t_dress_begin) \
            * (t_dress - t_dress_begin)

    DeltaRb_t_dress = DeltaRb_max + \
            (DeltaRb_min - DeltaRb_max) / (t_dress_end - t_dress_begin) \
            * (t_dress - t_dress_begin)

    OmegaRa_t_undress = OmegaR_min + (OmegaR_max - OmegaR_min) \
                * exp(-(t_undress - t_undress_begin) / 2 / t_gaussian_width**2)

    OmegaRb_t_undress = OmegaR_min + (OmegaR_max - OmegaR_min) \
                * exp(-(t_undress - t_undress_begin) / 2 / t_gaussian_width**2)

    DeltaRa_t_undress = DeltaRa_min + \
            (DeltaRa_max - DeltaRa_min) / (t_undress_end - t_undress_begin) \
                      * (t_undress - t_undress_begin)

    DeltaRb_t_undress = DeltaRb_min + \
            (DeltaRb_max - DeltaRb_min) / (t_undress_end - t_undress_begin) \
            * (t_undress - t_undress_begin)

    hamiltonians_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_dress = np.identity(Ndim)

    hamiltonians_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_undress = np.identity(Ndim)
    
    
    for n in range(Nsteps):
        h_params_dress = {\
            'OmegaRa' : OmegaRa_t_dress[n], \
            'OmegaRb' : OmegaRb_t_dress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa_t_dress[n], \
            'DeltaRb' : DeltaRb_t_dress[n], \
            'DeltaMWa' : DeltaMW, \
            'DeltaMWb' : DeltaMW, \
        }
        
        h_params_undress = {\
            'OmegaRa' : OmegaRa_t_undress[n], \
            'OmegaRb' : OmegaRb_t_undress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa_t_dress[n], \
            'DeltaRb' : DeltaRb_t_dress[n], \
            'DeltaMWa' : DeltaMW, \
            'DeltaMWb' : DeltaMW, \
        }

        hamiltonians_dress[:, :, n] = \
            rydbergatoms.hamiltonian_PerfectBlockade(0, h_params_dress)
        propagators_dress[:, :, n] = \
                        expm(-1j * Tstep * hamiltonians_dress[:, :, n])

        propagator_cumulative_dress = \
            np.dot(propagators_dress[:, :, n], propagator_cumulative_dress)
        
        hamiltonians_undress[:, :, n] = \
            rydbergatoms.hamiltonian_PerfectBlockade(0, h_params_undress)
        propagators_undress[:, :, n] = \
            expm(-1j * Tstep * hamiltonians_undress[:, :, n])

        propagator_cumulative_undress = \
            np.dot(propagators_undress[:, :, n], propagator_cumulative_undress)

    return propagator_cumulative_dress, propagator_cumulative_undress
        

def tdse_right_side (t, v, hamiltonian_function, hamiltonian_parameters):
    """
    Right side of the time dependent Schrodinger equation
    
    Parameters
    ----------
    t: time
    
    v: time evolution operator represented as a vector

    hamiltonian_function: function which computes the Hamiltonian
    matrix

    hamiltonian_parameters: parameters for the Hamiltonian function
    
    Returns
    -------
    v_rhs: right side of the time dependent Schrodinger equation 
    represented as a vector
    """

    h = hamiltonian_function(0, hamiltonian_parameters)

    n_dimensions = np.shape(h)[0]

    u = np.reshape(v, (n_dimensions, n_dimensions))

    u_rhs = -1j * np.dot(h, u)

    v_rhs = np.reshape(u_rhs, (n_dimensions * n_dimensions, ))

    return v_rhs
