#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  comparisonRydbergEntanglment_main.py
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
import rydbergatoms
import grape
import robustcostfunctions

from numpy import sqrt, pi, arctan2, cos, sin

def cnrs_protocol(hamiltonian_parameters):
	"""
	Parameters
	----------
	hamiltonian_parameters:
	Dictionary containing the parameters of the Hamiltonian
	
	Returns
	-------
	
	"""
	
	
