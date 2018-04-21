#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  randommatrices.py
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

import numpy as np
import scipy

def gen_randomUnitaryNearIdentity(d, epsilon=1/16):
    """
    Generates a random unitary matrix which is 
    very close to the identity by exponentiating
    a small parameter times a Hermitian matrix
    
    Parameters
    ----------
    d:
    Dimension of the vectors space on which the
    unitary matrix acts
    
    epsilon:
    Small number characterizing how close the 
    generated unitary is to the identity
    
    Returns
    -------
    u:
    Generated unitary matrix
    """
    
    real = np.random.normal(size=(d,d))
    imag = 1j*np.random.normal(size=(d,d))

    h = real + real.transpose() + imag + imag.transpose().conjugate()

    u = scipy.linalg.expm(-1j*epsilon*h)
    
    return u