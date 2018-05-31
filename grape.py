#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  grape.py
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
import scipy
import scipy.linalg
import scipy.optimize

from numpy import pi

def grape (control_problem, debug=False, gtol=1e-4):
    """
    Runs the Gradient Ascent Pulse Engineering Algorithm (GRAPE)
    to find a control sequence for quantum control
    
    Parameters
    ----------
    control_problem:
    Dictionary containing the parameters for the control problem being solved
    
    gtol:
    Minimum absolute value of gradient at which to stop
    """
    
    cost_function = control_problem.get('CostFunction')
    cost_function_grad = control_problem.get('CostFunctionGrad')
    Nsteps = control_problem.get('PropagatorParameters').get('Nsteps')
    
    
    if control_problem.get('Initialization') != None:
        initialization = control_problem['Initialization']
    else:
        initialization = 'Random'
        
    if initialization == 'Random':
        phi_initial = 2*pi * np.random.rand(Nsteps)
    elif initialization == 'Constant':
        phi_initial = np.zeros(Nsteps)
    elif initialization == 'Sine':
        phi_initial = pi * (1 + np.sin(2*pi*np.linspace(0, 2, Nsteps)))
        
    result = scipy.optimize.minimize(\
                                     fun=cost_function, x0=phi_initial, jac=True, method='BFGS', \
                                     options={'gtol': gtol, 'maxiter': 4096,}, \
                                     args=(control_problem,))

    phi_des = result.x
    infidelity_min = result.fun
    dinfidelity_min = result.jac
    status = result.status
    Niterations = result.nit

    #(phi_opt, infidelity_min, dinfidelity_opt, ddinfidelity_opt, func_calls, \
    # grad_calls, warnflag, allvects) = result
     
    if debug:
         print('# dinfidelity_min = %s\n # infidelity_min = %g\n' % (dinfidelity_min, infidelity_min))

    results = {\
               'PhiInitial': phi_initial, \
               'PhiOptimized' : phi_des, \
               'InfidelityMin' : infidelity_min, \
               'InfidelityMinGradient' : dinfidelity_min, \
               'Niterations' : Niterations, \
    }
    
    control_problem['Results'] = results
     
    return phi_des, infidelity_min
