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


import copy
import numpy as np
import scipy.linalg
import os, pickle

import datetime, time, uuid

from numpy import pi, sqrt, sin, cos, sign, exp, arctan2
from scipy.linalg import expm

OUTPUT_DIR = os.path.expanduser('~/RydbergGates/output/')

import adiabaticevolution
import grape
import rydbergatoms
import objectives

hamiltonian_base_parameters = {
    'OmegaR' : 2*pi * 4, \
    'OmegaMW' : 2*pi * 4, \
    'DeltaR' : 2*pi * 2, \
    'DeltaMW' : 0, \
}

hamiltonian_parameters = {
    'OmegaRa' : hamiltonian_base_parameters['OmegaR'], \
    'OmegaRb' : hamiltonian_base_parameters['OmegaR'], \
    'OmegaMWa' : hamiltonian_base_parameters['OmegaMW'], \
    'OmegaMWb' : hamiltonian_base_parameters['OmegaMW'], \
    'DeltaRa' : hamiltonian_base_parameters['DeltaR'], \
    'DeltaRb' : hamiltonian_base_parameters['DeltaR'], \
    'DeltaMWa' : hamiltonian_base_parameters['DeltaMW'], \
    'DeltaMWb' : hamiltonian_base_parameters['DeltaMW'], \
}


hamiltonian_landmarks_list = [\
    {\
        'DeltaRa': 2*pi * 2.1, \
        'DeltaRb': 2*pi * 2.1, \
    }, \
    {\
        'DeltaRa': 2*pi * 1.9, \
        'DeltaRb': 2*pi * 1.9, \
    }, \
    {\
        'DeltaRa': 2*pi * 2.1, \
        'DeltaRb': 2*pi * 1.9, \
    }, \
    {\
        'DeltaRa': 2*pi * 2.0, \
        'DeltaRb': 2*pi * 2.0, \
    }
]


Tpi = pi/hamiltonian_base_parameters['OmegaMW']
Nsteps_PiPulse = 8
N_PiPulses = 32
Nsteps = Nsteps_PiPulse * N_PiPulses
Tstep = Tpi / Nsteps_PiPulse
Tcontrol = Nsteps * Tstep 

propagator_parameters = {
    'HamiltonianParameters' : hamiltonian_parameters, \
    'HamiltonianMatrix' : rydbergatoms.hamiltonian_PerfectBlockade, \
    'HamiltonianMatrixGradient' : rydbergatoms.hamiltonian_grad_PerfectBlockade, \
    'Nsteps' : Nsteps, \
    'Tstep' : Tstep, \
    'Tcontrol' :  Tcontrol, \
}

u_target = rydbergatoms.ket_00 * rydbergatoms.bra_00 \
         - rydbergatoms.ket_01 * rydbergatoms.bra_01 \
         - rydbergatoms.ket_10 * rydbergatoms.bra_10 \
         - rydbergatoms.ket_11 * rydbergatoms.bra_11

control_problem = {
    'ControlTask' : 'UnitaryMap', \
    'Initialization' : 'Sine', \
    'UnitaryTarget': u_target, \
    'PropagatorParameters': propagator_parameters, \
    'CostFunction' : objectives.robust_adiabatic_infidelity_unitary, \
    'HamiltonianBaseParameters' : hamiltonian_base_parameters, \
    'HamiltonianLandmarks': hamiltonian_landmarks_list, \
    'InfidelityEvaluationInformation': [], \
}

adiabatic_parameters = {
    't_gaussian_width' : 0.25, \
    'DeltaMW' : 0, \
    'DeltaR_min': 2*pi * 2, \
    'DeltaR_max': 2*pi * 10, \
    'OmegaR_min': 0, \
    'OmegaR_max': 2*pi * 4, \
    'HamiltonianMatrix' : rydbergatoms.hamiltonian_PerfectBlockade, \
    'DimensionHilbertSpace': 8, \
    'Nsteps': 1024,
}

def add_adiabatic_landmark_unitaries(control_problem):
    u_dress_dict = {}
    u_undress_dict = {}
    u_dress_landmarks = []
    u_undress_landmarks = []

    DeltaR = hamiltonian_base_parameters['DeltaR']

    for hamiltonian_landmark_current in hamiltonian_landmarks_list:
        DeltaRa = hamiltonian_landmark_current.get('DeltaRa')
        DeltaRb = hamiltonian_landmark_current.get('DeltaRb')

        adiabatic_parameters_current = copy.deepcopy(adiabatic_parameters)
        adiabatic_parameters_current['DeltaRa_min'] = adiabatic_parameters['DeltaR_min'] + (DeltaRa - DeltaR)
        adiabatic_parameters_current['DeltaRa_max'] = adiabatic_parameters['DeltaR_max'] + (DeltaRa - DeltaR)
        adiabatic_parameters_current['DeltaRb_min'] = adiabatic_parameters['DeltaR_min'] + (DeltaRb - DeltaR)
        adiabatic_parameters_current['DeltaRb_max'] = adiabatic_parameters['DeltaR_max'] + (DeltaRb - DeltaR)

        u_dress, u_undress = \
            adiabaticevolution.adiabatic_evolution_propagators(adiabatic_parameters_current)

        u_dress_dict[(DeltaRa, DeltaRb)] = u_dress
        u_undress_dict[(DeltaRa, DeltaRb)] = u_undress
        u_dress_landmarks.append(u_dress)
        u_undress_landmarks.append(u_undress)

    control_problem['UnitaryDressing'] = u_dress_dict
    control_problem['UnitaryUndressing'] = u_undress_dict
    control_problem['UnitaryDressingLandmarks'] = u_dress_landmarks
    control_problem['UnitaryUnDressingLandmarks'] = u_undress_landmarks


def save_simulation(time_string):
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    filename = time_string + '--' + str(uuid.uuid4()) + '.pkl'
    outputfile = open(os.path.join(OUTPUT_DIR, filename), 'wb')

    simulation_data = {
        'time_string': time_string,
        'hamiltonian_parameters': hamiltonian_parameters,
        'hamiltonian_base_parameters': hamiltonian_base_parameters,
        'hamiltonian_landmarks_list': hamiltonian_landmarks_list,
        'control_problem': control_problem,
        'propagator_parameters': propagator_parameters,
    }

    pickle.dump(simulation_data, outputfile)
    outputfile.close()


def main():
    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Starting simulation for Tstep=%gns; Nsteps=%d, Tcontrol=%gns" %
          (time_string, Tstep*1e3, Nsteps, Tcontrol*1e3))

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Calculating adiabatic dressing and undressing unitaries" % time_string)
    add_adiabatic_landmark_unitaries(control_problem)

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Optimizing microwave phase forms" % time_string)
    grape.grape(control_problem, debug=True, gtol=1e-5)

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Saving simulation data" % time_string)

    time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')
    save_simulation(time_string)

    time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Finished" % time_string)


if __name__ == '__main__':
    main()
