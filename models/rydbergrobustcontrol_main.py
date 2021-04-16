#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergrobustcontrol_main.py
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
import rydbergatoms
import grape
import robustcostfunctions

from numpy import sqrt, pi, arctan2, cos, sin

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

hamiltonian_parameters = {
    'OmegaRa' : 1, \
    'OmegaRb' : 1, \
    'OmegaMWa' : 1, \
    'OmegaMWb' : 1, \
    'DeltaRa' : 0.01, \
    'DeltaRb' : 0.01, \
    'DeltaMWa' : 0.01, \
    'DeltaMWb' : 0.01, \
}

hamiltonian_base_parameters = {
    'OmegaR' : 1, \
    'OmegaMW' : 1, \
    'DeltaR' : 0.01, \
    'DeltaMW' : 0.01, \
}

hamiltonian_uncertain_parameters = {
    'deltaDeltaRValues' : [-1/20, -1/20]
}

Nsteps_PiPulse = 16

propagator_parameters = {
    'HamiltonianParameters' : hamiltonian_parameters, \
    'HamiltonianMatrix' : rydbergatoms.hamiltonian_PerfectBlockade, \
    'HamiltonianMatrixGradient' : rydbergatoms.hamiltonian_grad_PerfectBlockade, \
    'Nsteps' : 16 * Nsteps_PiPulse, \
    'Tstep' : pi / Nsteps_PiPulse, \
    'Tcontrol' : 16 * pi, \
}

control_problem = {
    'ControlTask' : 'StateToStateMap', \
    'Initialization' : 'Constant', \
    'PropagatorParameters': propagator_parameters, \
    'PureStateInitial': rydbergatoms.ket_00, \
    'PureStateTarget': calc_targetdressedstate(hamiltonian_parameters), \
    #'PureStateTarget': (rydbergatoms.ket_01 + rydbergatoms.ket_10)/sqrt(2), \
    'CostFunction' : robustcostfunctions.infidelity, \
    'CostFunctionGrad' : robustcostfunctions.infidelity_gradient, \
    #'CostFunctionGrad' : None, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianUncertainParameters' : hamiltonian_uncertain_parameters, \
}

if __name__ == '__main__':
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)