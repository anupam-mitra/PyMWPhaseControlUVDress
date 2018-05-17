#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergatoms.py
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
from numpy import exp


"""
Basis vectors for the eight dimensional Hilbert space
in the perfect blockade limit
"""
ket_r1 = np.asarray([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_1r = np.asarray([[0], [1], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_r0 = np.asarray([[0], [0], [1], [0], [0], [0], [0], [0]], dtype=complex)
ket_0r = np.asarray([[0], [0], [0], [1], [0], [0], [0], [0]], dtype=complex)
ket_11 = np.asarray([[0], [0], [0], [0], [1], [0], [0], [0]], dtype=complex)
ket_10 = np.asarray([[0], [0], [0], [0], [0], [1], [0], [0]], dtype=complex)
ket_01 = np.asarray([[0], [0], [0], [0], [0], [0], [1], [0]], dtype=complex)
ket_00 = np.asarray([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=complex)

bra_00 = ket_00.conjugate().transpose()
bra_01 = ket_01.conjugate().transpose()
bra_0r = ket_0r.conjugate().transpose()
bra_10 = ket_10.conjugate().transpose()
bra_11 = ket_11.conjugate().transpose()
bra_1r = ket_1r.conjugate().transpose()
bra_r0 = ket_r0.conjugate().transpose()
bra_r1 = ket_r1.conjugate().transpose()


def hamiltonian_TwoDress (h_params):
    """
    Computes the Hamiltonian matrix for the states that 
    participate in the two atom dressing.
    
    Paramters
    ---------
    h_params:
    Dictionary containing the parameters of the Hamiltonian
    
    Returns
    -------
    H_TwoDress:
    The hamiltonian matrix
    """
    DeltaRa = h_params['DeltaRa']
    DeltaRb = h_params['DeltaRb']
    OmegaRa = h_params['OmegaRa']
    OmegaRb = h_params['OmegaRb']
    DeltaMWa = h_params['DeltaMWa']
    DeltaMWb = h_params['DeltaMWb']
    OmegaMWa = h_params['OmegaMWa']
    OmegaMWb = h_params['OmegaMWb']
    
    H_TwoDress = -(DeltaRa + DeltaMWa + DeltaMWb) * ket_r1 * bra_r1 \
        -(DeltaRb + DeltaMWa + DeltaMWb) * ket_1r * bra_1r \
        -(DeltaMWa + DeltaMWb) * ket_11 * bra_11 \
        + OmegaRa / 2 * (ket_r1 * bra_11 + ket_11 * bra_r1) \
        + OmegaRb / 2 * (ket_1r * bra_11 + ket_11 * bra_1r)

    return H_TwoDress

def hamiltonian_OneDress (h_params):
    """
    Computes the Hamiltonian matrix for the states that 
    participate in the one atom dressing.
    
    Parameters
    ---------
    h_params:
    Dictionary containing the parameters of the Hamiltonian
    
    Returns
    -------
    H_OneDress:
    The hamiltonian matrix
    """
    DeltaRa = h_params['DeltaRa']
    DeltaRb = h_params['DeltaRb']
    OmegaRa = h_params['OmegaRa']
    OmegaRb = h_params['OmegaRb']
    DeltaMWa = h_params['DeltaMWa']
    DeltaMWb = h_params['DeltaMWb']
    OmegaMWa = h_params['OmegaMWa']
    OmegaMWb = h_params['OmegaMWb']

    H_OneDress = -(DeltaRa + DeltaMWa) * ket_r0 * bra_r0 \
        -(DeltaRb + DeltaMWb) * ket_0r * bra_0r \
        - DeltaMWa * ket_10 * bra_10 \
        - DeltaMWb * ket_01 * bra_01 \
        + OmegaRa / 2 * (ket_r0 * bra_10 + ket_10 * bra_r0) \
        + OmegaRb / 2 * (ket_0r * bra_01 + ket_01 * bra_0r)
        
    return H_OneDress
        
def hamiltonian_MW (phi, h_params):
    """
    Computes the Hamiltonian matrix for the microwave control
    
    Parameters
    ---------
    phi:
    Phase of the microwave
    
    h_params:
    Dictionary containing the parameters of the Hamiltonian
    
    Returns
    -------
    H_TwoDress:
    The hamiltonian matrix
    """
    DeltaRa = h_params['DeltaRa']
    DeltaRb = h_params['DeltaRb']
    OmegaRa = h_params['OmegaRa']
    OmegaRb = h_params['OmegaRb']
    DeltaMWa = h_params['DeltaMWa']
    DeltaMWb = h_params['DeltaMWb']
    OmegaMWa = h_params['OmegaMWa']
    OmegaMWb = h_params['OmegaMWb']
    
    H_MW = OmegaMWa / 2 * (exp(1j*phi) * ket_0r * bra_1r + exp(-1j*phi) * ket_1r * bra_0r) \
          + OmegaMWb / 2 * (exp(1j*phi) * ket_r0 * bra_r1 + exp(-1j*phi) * ket_r1 * bra_r0) \
          + OmegaMWa / 2 * (exp(1j*phi) * ket_01 * bra_11 + exp(-1j*phi) * ket_11 * bra_01) \
          + OmegaMWb / 2 * (exp(1j*phi) * ket_10 * bra_11 + exp(-1j*phi) * ket_11 * bra_10) \
          + OmegaMWa / 2 * (exp(1j*phi) * ket_00 * bra_10 + exp(-1j*phi) * ket_10 * bra_00) \
          + OmegaMWb / 2 * (exp(1j*phi) * ket_00 * bra_01 + exp(-1j*phi) * ket_01 * bra_00)

    return H_MW
        
def hamiltonian_PerfectBlockade (phi, hparams):
    """
    Computes the Hamiltonian matrix in the perfect
    blockade regime
    
    Parameters
    ---------
    phi:
    Phase of the microwave

    h_params:
    Dictionary containing the parameters of the Hamiltonian
    
    Returns
    -------
    H_PerfBlock:
    The hamiltonian matrix
    """
    H_TwoDress = hamiltonian_TwoDress(hparams)
    H_OneDress = hamiltonian_OneDress(hparams)
    H_MW = hamiltonian_MW(phi, hparams)
    H_PerfBlock = H_TwoDress + H_OneDress + H_MW
    
    return H_PerfBlock

def hamiltonian_grad_PerfectBlockade (phi, h_params):
    """
    Computes the derivative Hamiltonian matrix with respect
    to the phase of the microwave
    
    Parameters
    ---------
    phi:
    Phase of the microwave

    h_params:
    Dictionary containing the parameters of the Hamiltonian
    
    Returns
    -------
    dH_dphi:
    The derivative hamiltonian matrix
    """
    DeltaRa = h_params['DeltaRa']
    DeltaRb = h_params['DeltaRb']
    OmegaRa = h_params['OmegaRa']
    OmegaRb = h_params['OmegaRb']
    DeltaMWa = h_params['DeltaMWa']
    DeltaMWb = h_params['DeltaMWb']
    OmegaMWa = h_params['OmegaMWa']
    OmegaMWb = h_params['OmegaMWb']

    
    dH_dphi = 1j * OmegaMWa / 2 * (exp(1j*phi) * ket_0r * bra_1r - exp(-1j*phi) * ket_1r * bra_0r) \
          + 1j * OmegaMWb / 2 * (exp(1j*phi) * ket_r0 * bra_r1 - exp(-1j*phi) * ket_r1 * bra_r0) \
          + 1j * OmegaMWa / 2 * (exp(1j*phi) * ket_01 * bra_11 - exp(-1j*phi) * ket_11 * bra_01) \
          + 1j * OmegaMWb / 2 * (exp(1j*phi) * ket_10 * bra_11 - exp(-1j*phi) * ket_11 * bra_10) \
          + 1j * OmegaMWa / 2 * (exp(1j*phi) * ket_00 * bra_10 - exp(-1j*phi) * ket_10 * bra_00) \
          + 1j * OmegaMWb / 2 * (exp(1j*phi) * ket_00 * bra_01 - exp(-1j*phi) * ket_01 * bra_00)

    return dH_dphi
