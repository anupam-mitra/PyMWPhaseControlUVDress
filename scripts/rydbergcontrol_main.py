#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergcontrol_main.py
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
from control import rydbergcontrol

hamiltonian_parameters = rydbergcontrol.hamiltonian_parameters(
    DeltaR=2, DeltaMW=0.01, deltaRa=0.01, deltaRb=-0.01)

Nsteps_PiPulse = 1
propagator_parameters = rydbergcontrol.phase_propagator_parameters(
    hamiltonian_parameters, Nsteps_PiPulse, 16)
control_problem = rydbergcontrol.state_control_problem(
    hamiltonian_parameters, propagator_parameters)

if __name__ == '__main__':
    phi_opt, infidelity_min = grape.grape(control_problem, debug=True)
