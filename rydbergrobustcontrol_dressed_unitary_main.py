#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergrobust_control_dressed_unitary_main.py
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

import adiabaticevolution
import rydbergatoms
import grape
import robustcostfunctions

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
    'CostFunction' : robustcostfunctions.infidelity_unitary, \
    'CostFunctionGrad' : robustcostfunctions.infidelity_unitary_gradient, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianUncertainParameters' : hamiltonian_uncertain_parameters, \
}

if __name__ == '__main__':
	
	
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True, gtol=1e-5)
