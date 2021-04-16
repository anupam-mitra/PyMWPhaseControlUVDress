#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  utilities.py
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

import numpy as np

def check_unitary(u):
    """
    Checks if a matrix is unitary
    
    Parameters
    ----------
    u:
    Matrix to be checked for unitarity
    
    Returns
    -------
    norm:
    Frobenius norm of udagger * u
    """
    
    udagger = u.transpose().conjugate()
    Ndims = u.shape[0]
    norm = np.linalg.trace(np.real(np.dot(udagger, u) + np.dot(u, udagger))) / (2*Ndims)
    
    return norm

def check_hermitian(h):
    """
    Checks if a matrix is hermitian
    
    Parameters
    ----------
    u:
    Matrix to be checked for unitarity
    
    Returns
    -------
    norm:
    Frobenius norm of hdagger - h
    """
    
    hdagger = h.transpose().conjugate()
    Ndims = h.shape[0]
    norm = np.linalg.norm(hdagger - h) / (2*Ndims)
    
    return norm

dagger = lambda u : np.transpose(np.conjugate(u))

def calc_hilbertschmidt_innerproduct (s, t):
    """
    Calculates the Hilbert Schmidt inner product
    between two matrices 

    Parameters
    ----------
    s, t: 
    The matrices for which to compute the Hilbert
    Schmidt inner product

    Returns:
    -------
    hs:
    The Hilbert Schmidt inner product of the two
    matrices
    """

    hs = np.dot(s, dagger(t))
    return hs
