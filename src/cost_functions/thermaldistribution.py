#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  thermaldistribution.py
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

from numpy import exp, sqrt, pi

class ThermalDistribution:
    """
    Represents the thermal distribution of velocities and detuning for atoms
    which are free particles, that is the Maxwell Boltzmann distribution

    Parameters
    ----------
    natoms: int
    Number of atoms

    temperature: float
    Temperature of the atoms

    k_Boltzmann:
    Boltzmann's constant in the units of the problem

    m_atom:
    Mass of the atoms in the units of the problem
    """

    k_Boltzmann_SI = 1.38064852e-23 # J/K
    m_atom_SI = 2.20694650e-25 # kg, for Cesium

    def __init__ (self, natoms, temperature, m_atom=1):
        self.natoms = natoms
        self.temperature = temperature
        self.m_atom = m_atom

    def calc_MaxwellBoltzmann (v, temperature=None):
        """
        Calculates the probability density function for the Maxwell Boltzmann
        distribution

        Parameters
        ----------
        v: ndarray <float> (natoms)
        Velocities of the atoms

        temperature: float
        Temperature
        """

        if temperature == None:
            temperature = self.temperature

        m_atom = self.m_atom
        k_Boltzmann = self.k_Boltzmann

        v_variance = k_Boltzmann * temperature / m_atom

        p = sqrt(1 / (2 * pi * v_variance)) ** natoms
        for a in range(natoms):
            p = p * exp( - 1/2 * (v / v_variance)**2)

        return p
