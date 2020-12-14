#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergcontrol_unitary_main.py
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

import numpy as np
import rydbergatoms
import grape
import costfunctions

from numpy import sqrt, pi, arctan2, exp, cos, sin, sign

def calc_targetdressedstate (hamiltonian_parameters):
    """
    Finds the target dressed state
    
    Parameters
    ----------
    hamiltonian_parameters:
        Dictionary containing the parameters of the Hamiltonian

    Returns
    -------
    psi_target:
        Target state
    """
    
    DeltaRa = hamiltonian_parameters['DeltaRa']
    DeltaRb = hamiltonian_parameters['DeltaRb']
    OmegaRa = hamiltonian_parameters['DeltaRa']
    OmegaRb = hamiltonian_parameters['DeltaRb']
    
    thetaMixa = arctan2(OmegaRa, -DeltaRa)
    thetaMixb = arctan2(OmegaRb, -DeltaRb)
    
    psi_target = 1/sqrt(2)* \
               ( cos(thetaMixa/2) * rydbergatoms.ket_r0 \
               + sin(thetaMixa/2) * rydbergatoms.ket_10 \
               + cos(thetaMixb/2) * rydbergatoms.ket_0r \
               + sin(thetaMixb/2) * rydbergatoms.ket_01 )

    return psi_target

def calc_dressingunitary (hamiltonian_parameters):
    """
    Calculates the unitary map implemented during the adiabatic Rydberg
    dressing. The dressing time is chosen such that the difference in accumulation
    of phase due to the one atom light shift and two atom light shifts is pi/2
    
    Parameters
    ----------
    hamiltonian_parameters:
        Dictionary containing the parameters of the Hamiltonian
        
    Returns
    -------
    udress:
        Unitary map implemented during the adiabatic Rydberg dressing
    """

    DeltaRa = hamiltonian_parameters['DeltaRa']
    DeltaRb = hamiltonian_parameters['DeltaRb']
    OmegaRa = hamiltonian_parameters['DeltaRa']
    OmegaRb = hamiltonian_parameters['DeltaRb']
    
    DeltaR = (DeltaRa + DeltaRb)/2
    OmegaR = (OmegaRa + OmegaRb)/2
    
    ELS1 = -DeltaR/2 + sign(DeltaR) * sqrt(DeltaR**2 + OmegaR**2)
    ELS2 = -DeltaR/2 + sign(DeltaR) * sqrt(DeltaR**2 + 2*OmegaR**2)
    
    Tdress = pi/(ELS2 - ELS1)/2
    
    udress = rydbergatoms.ket_00 * rydbergatoms.bra_00 \
           + exp(-1j * ELS1 * Tdress) * rydbergatoms.ket_01 * rydbergatoms.bra_01 \
           + exp(-1j * ELS1 * Tdress) * rydbergatoms.ket_10 * rydbergatoms.bra_10 \
           + exp(-1j * ELS2 * Tdress) * rydbergatoms.ket_11 * rydbergatoms.bra_11 \
           + exp(+1j * ELS1 * Tdress) * rydbergatoms.ket_0r * rydbergatoms.bra_0r \
           + exp(+1j * ELS1 * Tdress) * rydbergatoms.ket_r0 * rydbergatoms.bra_r0 \
           + exp(+1j * ELS2 * Tdress) \
           * (rydbergatoms.ket_r1 + rydbergatoms.ket_1r)/sqrt(2) \
           * (rydbergatoms.bra_r1 + rydbergatoms.bra_1r)/sqrt(2) \
           + (rydbergatoms.ket_r1 - rydbergatoms.ket_1r)/sqrt(2) \
           * (rydbergatoms.bra_r1 - rydbergatoms.bra_1r)/sqrt(2) \

    return udress

hamiltonian_parameters = {
    'OmegaRa' : 1, \
    'OmegaRb' : 1, \
    'OmegaMWa' : 1, \
    'OmegaMWb' : 1, \
    'DeltaRa' : 2 + 0.01, \
    'DeltaRb' : 2 - 0.01, \
    'DeltaMWa' : 0.01, \
    'DeltaMWb' : 0.01, \
}

Nsteps_PiPulse = 32

propagator_parameters = {
    'HamiltonianParameters' : hamiltonian_parameters, \
    'HamiltonianMatrix' : rydbergatoms.hamiltonian_PerfectBlockade, \
    'HamiltonianMatrixGradient' : rydbergatoms.hamiltonian_grad_PerfectBlockade, \
    'Nsteps' : 64 * Nsteps_PiPulse, \
    'Tstep' : pi / Nsteps_PiPulse, \
    'Tcontrol' : 64 * pi, \
}

u_target = rydbergatoms.ket_00 * rydbergatoms.bra_00 \
         + rydbergatoms.ket_01 * rydbergatoms.bra_01 \
         + rydbergatoms.ket_10 * rydbergatoms.bra_10 \
         - rydbergatoms.ket_11 * rydbergatoms.bra_11
         
u_dress = calc_dressingunitary(hamiltonian_parameters)
#u_dress = np.identity(8)

control_problem = {
    'ControlTask' : 'UnitaryMap', \
    'Initialization' : 'Random', \
    'UnitaryTarget' : np.dot(u_dress, np.dot(u_target, u_dress.transpose().conjugate())), \
    'PropagatorParameters': propagator_parameters, \
    'CostFunction' : costfunctions.infidelity_unitary, \
    'CostFunctionGrad' : costfunctions.infidelity_unitary_gradient \
}

if __name__ == '__main__':
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
