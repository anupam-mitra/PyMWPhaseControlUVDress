#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  adibaticevolution.py
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

import numpy as np
import scipy

from scipy.linalg import expm

import rydbergatoms

def compute_adiabaticRydbergDressing_propagator(adiabatic_params):
    """
    Computes the propagator for an adiabatic evolution where the 
    Hamiltonian is adiabatically changed from one Hamiltonian to another

    Parameters
    ----------
            
    adiabatic_params:
        Parameters to use for adiabatic evolution
    
    """
    
    Nsteps = adiabatic_params.get('Nsteps')
    Tstep = adiabatic_params.get('Tstep')
    Tcontrol = adiabatic_params.get('Tcontrol')
    Ndim = adiabatic_params.get('DimensionHilbertSpace')
    
    hamiltonian_func = adiabatic_params.get('HamiltonianMatrix')
    hamiltonian_grad_func = adiabatic_params.get('HamiltonianMatrixGradient')
    
    h_params = propagator_params.get('HamiltonianParameters')

    hamiltonians_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_dress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_dress = np.identity(Nsteps)

    hamiltonians_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagators_undress = np.empty((Ndim, Ndim, Nsteps), dtype=complex)
    propagator_cumulative_undress = np.identity(Nsteps)

    t_dress_initial = -8*Tsweep
    t_dress_final = 0
    t_undress_initial = 0
    t_dress_final = +8*Tsweep
    
    t_dress = np.arange(t_dress_initial, t_dress_final + Tstep, Tstep)
    t_undress = np.arange(t_undress_initial, t_undress_final + Tstep, Tstep)

    DeltaRa_dress = DeltaRa * (2 - 1*exp(-(t_dress/Tsweep)**2/2))
    OmegaRa_dress = OmegaRa * (0 + 1*exp(-(t_dress/Tsweep)**2/2))

    DeltaRb_dress = DeltaRb * (2 - 1*exp(-(t_dress/Tsweep)**2/2))
    OmegaRb_dress = OmegaRb * (0 + 1*exp(-(t_dress/Tsweep)**2/2))

    DeltaRa_undress = DeltaRa * (2 - 1*exp(-(t_undress/Tsweep)**2/2))
    OmegaRa_undress = OmegaRa * (0 + 1*exp(-(t_undress/Tsweep)**2/2))

    DeltaRb_undress = DeltaRb * (2 - 1*exp(-(t_undress/Tsweep)**2/2))
    OmegaRb_undress = OmegaRb * (0 + 1*exp(-(t_undress/Tsweep)**2/2))


    for n in range(Nsteps):
        h_params_dress = {\
            'OmegaRa' : OmegaRa_dress[n], \
            'OmegaRb' : OmegaRb_dress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa_dress[n], \
            'DeltaRb' : DeltaRa_dress[b], \
            'DeltaMWa' : 0.01, \
            'DeltaMWb' : 0.01, \
        }
        
        h_params_undress = {\
            'OmegaRa' : OmegaRa_undress[n], \
            'OmegaRb' : OmegaRb_undress[n], \
            'OmegaMWa' : 0, \
            'OmegaMWb' : 0, \
            'DeltaRa' : DeltaRa_undress[n], \
            'DeltaRb' : DeltaRa_undress[b], \
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

        propagator_cumulative_undress = np.dot(propagators_dress[:, :, n], propagator_cumulative_undress)


    return propagator_cumulative_dress, propagator_cumulative_undress
    
