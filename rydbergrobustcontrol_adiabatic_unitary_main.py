#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergrobust_control_dressed_unitary_main.py
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
import robustadiabaticcostfunctions

from numpy import sqrt, pi, arctan2, cos, sin

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

Nsteps_PiPulse = 4

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

control_problem = {
    'ControlTask' : 'UnitaryMap', \
    'Initialization' : 'Random', \
    'Initialization' : 'Constant', \
    'UnitaryTarget': u_target, \
    'PropagatorParameters': propagator_parameters, \
    'CostFunction' : robustadiabaticcostfunctions.infidelity_unitary, \
    'CostFunctionGrad' : robustadiabaticcostfunctions.infidelity_unitary_gradient, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianUncertainParameters' : hamiltonian_uncertain_parameters, \
}

adiabatic_parameters = {
    'DimensionHilbertSpace' : 8,\
    'Tstep' : pi / 16, \
    'TSweepFactor' : 16, \
    'tinitial': - 1024*pi, \
    'tfinal': + 1024*pi, \
    'HamiltonianMatrix' : rydbergatoms.hamiltonian_PerfectBlockade,
    'HamiltonianParameters' : hamiltonian_parameters,
}

if __name__ == '__main__':
    # Calculated dressing unitary for different parameters
    deltaR_values = hamiltonian_uncertain_parameters.get('deltaDeltaRValues')
    DeltaR = hamiltonian_base_parameters.get('DeltaR')

    u_dress_dict = {}
    u_undress_dict = {}
    
    for deltaRa, deltaRb in itertools.product(deltaR_values, deltaR_values):
        
        adiabatic_parameters['HamiltonianParameters']['DeltaRa'] \
        = DeltaR + deltaRa
        adiabatic_parameters['HamiltonianParameters']['DeltaRb'] \
        = DeltaR + deltaRb
    
        u_dress, u_undress = \
            adiabaticevolution.adiabaticrydbergdressing_propagator_detuningsweep(adiabatic_parameters)

        u_dress_dict[(deltaRa, deltaRb)] = u_dress
        u_undress_dict[(deltaRa, deltaRb)] = u_undress
    
    control_problem['UnitaryDressing'] =  u_dress_dict
    control_problem['UnitaryUndressing'] = u_undress_dict
    
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True, gtol=1e-5)
