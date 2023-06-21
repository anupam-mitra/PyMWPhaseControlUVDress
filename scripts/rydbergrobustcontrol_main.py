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


import control.optimization as grape
import rydbergcontrol

hamiltonian_base_parameters = rydbergcontrol.base_hamiltonian_parameters()
hamiltonian_parameters = rydbergcontrol.hamiltonian_parameters()

hamiltonian_uncertain_parameters = {
    'deltaDeltaRValues' : [-1/20, -1/20]
}

Nsteps_PiPulse = 16
propagator_parameters = rydbergcontrol.phase_propagator_parameters(
    hamiltonian_parameters, Nsteps_PiPulse, 16)
control_problem = rydbergcontrol.state_control_problem(
    hamiltonian_parameters, propagator_parameters, robust=True,
    base_params=hamiltonian_base_parameters,
    uncertain_params=hamiltonian_uncertain_parameters)

if __name__ == '__main__':
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
