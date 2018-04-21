#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  oneaxisttwistrotate_main.py
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

import oneaxistwistrotate
import grape
import costfunctions
import randommatrices

from numpy import sqrt, pi

s = 3/2
Ndim = int(2*s+1)
u_target = randommatrices.gen_randomUnitaryNearIdentity(Ndim, epsilon=1/1024)
#u_target = np.identity(Ndim)

hamiltonian_parameters = {
    'Omega' : 1, \
    's' : s, \
}

Nsteps_PiPulse =  128

propagator_parameters = {
    'HamiltonianParameters' : hamiltonian_parameters, \
    'HamiltonianMatrix' : oneaxistwistrotate.hamiltonian_SpinSymmetric, \
    'HamiltonianMatrixGradient' : oneaxistwistrotate.hamiltonian_grad_SpinSymmetric, \
    'Nsteps' : Nsteps_PiPulse, \
    'Tstep' : pi / Nsteps_PiPulse * Ndim, \
    'Tcontrol' : pi, \
}

control_problem = {
    'ControlTask' : 'UnitaryMap', \
    'Initialization' : 'Sine', \
    'UnitaryTarget' : u_target, \
    'PropagatorParameters': propagator_parameters, \
    'CostFunction' : costfunctions.infidelity_unitary, \
    'CostFunctionGrad' : costfunctions.infidelity_unitary_gradient, \
    #'CostFunctionGrad' : None, \
}

if __name__ == '__main__':
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
    
else:
    Niterations = 2
    overlaps = np.empty(Niterations)
    infidelities_design = np.empty(Niterations)
    for i in range(0, Niterations):
        print('Iteration %d' % i)
        
        u_target = randommatrices.gen_randomUnitaryNearIdentity(Ndim, epsilon=1)
        overlap_with_identity = np.abs(np.trace(u_target))**2/(u_target.shape[0])**2
        overlaps[i] = overlap_with_identity

        control_problem['UnitaryTarget'] = u_target        
        phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
        infidelities_design[i] = infidelity_min