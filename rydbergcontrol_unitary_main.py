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

from numpy import pi

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
         
u_dress = rydbergatoms.dressing_unitary(hamiltonian_parameters)
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
