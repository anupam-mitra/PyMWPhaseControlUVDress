#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  oneaxistwistrotate.py
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
from numpy import sqrt, sin, cos

def gen_AngularMomentum (s):
    """
    Generates angular momentum matrices for spin s
    
    Parameters
    ----------
    s:
    Spin quantum number
    
    Returns
    -------
    Sx:
    Matrix for the x component of the angular momentum

    Sy:
    Matrix for the y component of the angular momentum

    Sz:
    Matrix for the z component of the angular momentum
    """
    
    Sz = np.diag(np.arange(-s, s+1))
    eigenvalues = np.arange(-s, s+1)

    d = int(2*s) + 1
    I = np.identity(d)

    Splus = np.zeros((d, d))
    Sminus = np.zeros((d, d))

    for m in range(d - 1):
        splusfactor = sqrt(s*(s + 1) - eigenvalues[m]*(eigenvalues[m] + 1))
        sminusfactor = sqrt(s*(s + 1) - eigenvalues[m+1]*(eigenvalues[m+1] - 1))
        Splus = Splus + splusfactor *  np.outer(I[m, :], I[m+1,:])
        Sminus = Sminus + sminusfactor * np.outer(I[m+1, :], I[m, :])

    Sx = 1/2 * (Splus + Sminus)
    Sy = -1j/2 * (Splus - Sminus)

    return Sx, Sy, Sz

def hamiltonian_SpinSymmetric(phi, h_params):
    """
    Generates the one axis twisting and rotating Hamiltonian,
    which rotates about an axis making angle phi with the x 
    axis
    
    Parameters
    ----------
    phi:
    Angle of the rotation axis with the x axis
    
    h_params:
    Parameters of the Hamiltonian
    
    Returns
    -------
    H_TwistRotate:
    Hamiltonian matrix
    """
    
    s = h_params['s']
    Omega = h_params['Omega']
    Sx, Sy, Sz = gen_AngularMomentum(s)
    
    H_TwistRotate = np.dot(Sz, Sz) + Omega*(Sx * cos(phi) + Sy * sin(phi))
    
    return H_TwistRotate

def hamiltonian_grad_SpinSymmetric(phi, h_params):
    """
    Generates the one axis twisting and rotating Hamiltonian,
    which rotates about an axis making angle phi with the x 
    axis
    
    Parameters
    ----------
    phi:
    Angle of the rotation axis with the x axis
    
    h_params:
    Parameters of the Hamiltonian
    
    Returns
    -------
    H_TwistRotate:
    Hamiltonian matrix
    """
    
    s = h_params['s']
    Omega = h_params['Omega']
    Sx, Sy, Sz = gen_AngularMomentum(s)
    
    dH_dphi =  Omega*(-Sx * sin(phi) + Sy * cos(phi))
    
    return dH_dphi