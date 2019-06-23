#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rydbergatoms.py
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

from numpy import exp

from utilities import dagger

"""
Global TODO
===========


- Rename the classes to represent the name of the systems
- Find a solution to have a representation independent way of generating
operators, in particular, have an object with computes different kets and 
and bras for the system
- Design classes which perform adiabatic ramping up and ramping down
- Make the representation of atomic states \ket{0}, \ket{1} and \ket{r}
independent of the parameters of the problem


"""

class HamiltonianBase:
    """
    Represents a generic Hamiltonian for two atoms
    with two ground energy levels and a Rydberg energy
    level

    """
    # Basis vectors for the eight dimensional Hilbert space
    # in the perfect blockade limit

    ket_00 = np.asarray([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=complex)
    ket_01 = np.asarray([[0], [1], [0], [0], [0], [0], [0], [0]], dtype=complex)
    ket_10 = np.asarray([[0], [0], [1], [0], [0], [0], [0], [0]], dtype=complex)
    ket_11 = np.asarray([[0], [0], [0], [1], [0], [0], [0], [0]], dtype=complex)
    ket_0r = np.asarray([[0], [0], [0], [0], [1], [0], [0], [0]], dtype=complex)
    ket_r0 = np.asarray([[0], [0], [0], [0], [0], [1], [0], [0]], dtype=complex)
    ket_1r = np.asarray([[0], [0], [0], [0], [0], [0], [1], [0]], dtype=complex)
    ket_r1 = np.asarray([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=complex)

    bra_00 = ket_00.conjugate().transpose()
    bra_01 = ket_01.conjugate().transpose()
    bra_0r = ket_0r.conjugate().transpose()
    bra_10 = ket_10.conjugate().transpose()
    bra_11 = ket_11.conjugate().transpose()
    bra_1r = ket_1r.conjugate().transpose()
    bra_r0 = ket_r0.conjugate().transpose()
    bra_r1 = ket_r1.conjugate().transpose()

class HamiltonianConstantUV (HamiltonianBase):
    """
    Represents a Hamiltonian for two atoms with
    two ground energy levels 0, 1 and a Rydberg energy
    level r where
    - Ground to Rydberg coupling (1 to r, ultraviolet) has a
      fixed Rabi frequency and detuning
    - Ground coupling (0 to 1, microwave) has a fixed
      Rabi frequency and detuning
      HamiltonianBase

    Parameters
    ----------
    OmegaUV_a:
    Ultraviolet Rabi frequency for atom a

    OmegaUV_b:
    Ultraviolet Rabi frequency for atom b

    DeltaUV_a:
    Ultraviolet detuning for atom a

    DeltaUV_b:
    Ultraviolet detuning for atom b

    OmegaMW_a:
    Microwave Rabi frequency for atom a

    OmegaMW_b:
    Microwave Rabi frequency for atom b

    DeltaMW_a
    Microwave detuning for atom a

    DeltaMW_b
    Microwave detuning for atom b
    """
    def __init__ (self, OmegaUV_a, OmegaUV_b, DeltaUV_a, DeltaUV_b, \
                  OmegaMW_a, OmegaMW_b, DeltaMW_a, DeltaMW_b):
        self.OmegaUV_a = OmegaUV_a
        self.OmegaUV_b = OmegaUV_b
        self.DeltaUV_a = DeltaUV_a
        self.DeltaUV_b = DeltaUV_b
        self.OmegaMW_a = OmegaMW_a
        self.OmegaMW_b = OmegaMW_b
        self.DeltaMW_a = DeltaMW_a
        self.DeltaMW_b = DeltaMW_b

    def get_Hamiltonian_onedress_matrix (self):
        """
        Computes the part of the Hamiltonian
        which has states where each atom couples
        to its Rydberg state independently of the other.

        Returns
        -------
        H_onedress: Hamiltonian matrix
        """
        H_onedress = -(self.DeltaUV_a + self.DeltaMW_a) * np.outer(self.ket_r0, self.bra_r0) \
        - (self.DeltaUV_b + self.DeltaMW_b) * np.outer(self.ket_0r, self.bra_0r) \
        - self.DeltaMW_a * np.outer(self.ket_10, self.bra_10) \
        - self.DeltaMW_b * np.outer(self.ket_01, self.bra_01) \
        + self.OmegaUV_a / 2 * (np.outer(self.ket_r0, self.bra_10) + np.outer(self.ket_10, self.bra_r0)) \
        + self.OmegaUV_b/ 2 * (np.outer(self.ket_0r, self.bra_01) + np.outer(self.ket_01, self.bra_0r))

        return H_onedress

    def get_Hamiltonian_twodress_matrix (self):
        """
        Computes the part of the Hamiltonian
        which has states where the atom couple together
        to their Rydberg state.

        Returns
        -------
        H_onedress: Hamiltonian matrix
        """
        H_twodress = \
        -(self.DeltaUV_a + self.DeltaMW_a + self.DeltaMW_b) * np.outer(self.ket_r1, self.bra_r1) \
        -(self.DeltaUV_b + self.DeltaMW_a + self.DeltaMW_b) * np.outer(self.ket_1r, self.bra_1r) \
        -(self.DeltaMW_a + self.DeltaMW_b) * np.outer(self.ket_11, self.bra_11) \
        + self.OmegaUV_a / 2 * (np.outer(self.ket_r1, self.bra_11) + np.outer(self.ket_11, self.bra_r1)) \
        + self.OmegaUV_b / 2 * (np.outer(self.ket_1r, self.bra_11) + np.outer(self.ket_11, self.bra_1r))

        return H_twodress



class HamiltonianMWPhaseControl (HamiltonianConstantUV):
    """
    Represents a Hamiltonian for two atoms with
    two ground energy levels 0, 1 and a Rydberg energy
    level r where
    - Ground to Rydberg coupling (1 to r, ultraviolet) has a
      fixed Rabi frequency and detuning
    - Ground coupling (0 to 1, microwave) has a fixed
      Rabi frequency and detuning
    - Ground to ground coupling (microwave) has a controllable
     phase

    Parameters
    ----------
    OmegaUV_a:
    Ultraviolet Rabi frequency for atom a

    OmegaUV_b:
    Ultraviolet Rabi frequency for atom b

    DeltaUV_a:
    Ultraviolet detuning for atom a

    DeltaUV_b:
    Ultraviolet detuning for atom b

    OmegaMW_a:
    Microwave Rabi frequency for atom a

    OmegaMW_b:
    Microwave Rabi frequency for atom b

    DeltaMW_a
    Microwave detuning for atom a

    DeltaMW_b
    Microwave detuning for atom b
    """

    def __init__ (self, OmegaUV_a, OmegaUV_b, DeltaUV_a, DeltaUV_b, \
                  OmegaMW_a, OmegaMW_b, DeltaMW_a, DeltaMW_b):
        self.OmegaUV_a = OmegaUV_a
        self.OmegaUV_b = OmegaUV_b
        self.DeltaUV_a = DeltaUV_a
        self.DeltaUV_b = DeltaUV_b
        self.OmegaMW_a = OmegaMW_a
        self.OmegaMW_b = OmegaMW_b
        self.DeltaMW_a = DeltaMW_a
        self.DeltaMW_b = DeltaMW_b


    def get_Hamiltonian_drift (self):
        """
        Computes the part of the Hamiltonian which is not
        controllable

        Returns
        -------
        H_drift: Drift Hamiltonian matrix
        """
        H_onedress = self.get_Hamiltonian_onedress_matrix()
        H_twodress = self.get_Hamiltonian_twodress_matrix()
        H_drift = H_onedress + H_twodress
        return H_drift

    def get_Hamiltonian_mw (self, phi):
        """
        Computes the Hamiltonian matrix for the microwave control

        Parameters
        ---------
        phi:
        Phase of the microwave

        Returns
        -------
        H_mw:
        The hamiltonian matrix
        """
        H_mw = self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_0r, self.bra_1r) \
                                   + exp(+1j*phi) * np.outer(self.ket_1r, self.bra_0r))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_r0, self.bra_r1) \
                                    + exp(+1j*phi) * np.outer(self.ket_r1, self.bra_r0))\
              + self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_01, self.bra_11)\
                                    + exp(+1j*phi) * np.outer(self.ket_11, self.bra_01))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_10, self.bra_11)\
                                    + exp(+1j*phi) * np.outer(self.ket_11, self.bra_10))\
              + self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_00, self.bra_10)\
                                    + exp(+1j*phi) * np.outer(self.ket_10, self.bra_00))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_00, self.bra_01)\
                                    + exp(+1j*phi) * np.outer(self.ket_01, self.bra_00))

        return H_mw


    def get_Hamiltonian_control (self, phi):
        """
        Computes the control Hamitonian matrix for the microwave control

        Parameters
        ---------
        phi:
        Phase of the microwave

        Returns
        -------
        H_control:
        The hamiltonian matrix
        """

        H_control = self.get_Hamiltonian_mw(phi)
        return H_control

    def get_Hamiltonian_control_gradient (self, phi):
        """
        Computes the gradient of the Hamiltonian matrix with respect
        to the microwave phase for microwave control

        Parameters
        ----------
        phi:
        Phase of the microwave

        Returns:
        -------
        dHcontrol_dphi:
        Derivative of the Hamiltonian matrix with respect to
        the microwave phase
        """

        pass

class HamiltonianMWDetuningControl (HamiltonianConstantUV):
    """
    Represents a Hamiltonian for two atoms with
    two ground energy levels 0, 1 and a Rydberg energy
    level r where
    - Ground to Rydberg coupling (1 to r, ultraviolet) has a
      fixed Rabi frequency and detuning
    - Ground coupling (0 to 1, microwave) has a fixed
      Rabi frequency and detuning
    - Ground to ground coupling (microwave) has a controllable
     phase

    Parameters
    ----------
    OmegaUV_a:
    Ultraviolet Rabi frequency for atom a

    OmegaUV_b:
    Ultraviolet Rabi frequency for atom b

    DeltaUV_a:
    Ultraviolet detuning for atom a

    DeltaUV_b:
    Ultraviolet detuning for atom b

    OmegaMW_a:
    Microwave Rabi frequency for atom a

    OmegaMW_b:
    Microwave Rabi frequency for atom b

    DeltaMW_a
    Microwave detuning for atom a

    DeltaMW_b
    Microwave detuning for atom b
    """

    def __init__ (self, OmegaUV_a, OmegaUV_b, DeltaUV_a, DeltaUV_b, \
                  OmegaMW_a, OmegaMW_b, DeltaMW_a, DeltaMW_b):
        self.OmegaUV_a = OmegaUV_a
        self.OmegaUV_b = OmegaUV_b
        self.DeltaUV_a = DeltaUV_a
        self.DeltaUV_b = DeltaUV_b
        self.OmegaMW_a = OmegaMW_a
        self.OmegaMW_b = OmegaMW_b
        self.DeltaMW_a = DeltaMW_a
        self.DeltaMW_b = DeltaMW_b

    def get_Hamiltonian_drift (self):
        """
        Computes the part of the Hamiltonian which is not
        controllable

        Returns
        -------
        H_drift: Drift Hamiltonian matrix
        """
        H_onedress = self.get_Hamiltonian_onedress_matrix()
        H_twodress = self.get_Hamiltonian_twodress_matrix()
        H_drift = H_onedress + H_twodress
        return H_drift

    def get_Hamiltonian_mw (self, phi):
        """
        Computes the Hamiltonian matrix for the microwave control

        Parameters
        ---------
        phi:
        Phase of the microwave

        Returns
        -------
        H_mw:
        The hamiltonian matrix
        """
        H_mw = self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_0r, self.bra_1r) \
                                   + exp(+1j*phi) * np.outer(self.ket_1r, self.bra_0r))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_r0, self.bra_r1) \
                                    + exp(+1j*phi) * np.outer(self.ket_r1, self.bra_r0))\
              + self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_01, self.bra_11)\
                                    + exp(+1j*phi) * np.outer(self.ket_11, self.bra_01))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_10, self.bra_11)\
                                    + exp(+1j*phi) * np.outer(self.ket_11, self.bra_10))\
              + self.OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(self.ket_00, self.bra_10)\
                                    + exp(+1j*phi) * np.outer(self.ket_10, self.bra_00))\
              + self.OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(self.ket_00, self.bra_01)\
                                    + exp(+1j*phi) * np.outer(self.ket_01, self.bra_00))

        return H_mw


    def get_Hamiltonian_control (self, phi):
        """
        Computes the control Hamitonian matrix for the microwave control

        Parameters
        ---------
        phi:
        Phase of the microwave

        Returns
        -------
        H_control:
        The hamiltonian matrix
        """

        H_control = self.get_Hamiltonian_mw(phi)
        return H_control

    def get_Hamiltonian_control_gradient (self, phi):
        """
        Computes the gradient of the Hamiltonian matrix with respect
        to the microwave phase for microwave control

        Parameters
        ----------
        phi:
        Phase of the microwave

        Returns:
        -------
        dHcontrol_dphi:
        Derivative of the Hamiltonian matrix with respect to
        the microwave phase
        """

        pass




"""
Basis vectors for the eight dimensional Hilbert space
in the perfect blockade limit
"""
ket_r1 = np.asarray([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=complex)
ket_1r = np.asarray([[0], [0], [0], [0], [0], [0], [1], [0]], dtype=complex)
ket_r0 = np.asarray([[0], [0], [0], [0], [0], [1], [0], [0]], dtype=complex)
ket_0r = np.asarray([[0], [0], [0], [0], [1], [0], [0], [0]], dtype=complex)
ket_11 = np.asarray([[0], [0], [0], [1], [0], [0], [0], [0]], dtype=complex)
ket_10 = np.asarray([[0], [0], [1], [0], [0], [0], [0], [0]], dtype=complex)
ket_01 = np.asarray([[0], [1], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_00 = np.asarray([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=complex)

bra_00 = dagger(ket_00)
bra_01 = dagger(ket_01)
bra_0r = dagger(ket_0r)
bra_10 = dagger(ket_10)
bra_11 = dagger(ket_11)
bra_1r = dagger(ket_1r)
bra_r0 = dagger(ket_r0)
bra_r1 = dagger(ket_r1)

projector_logical = \
    np.dot(ket_00, bra_00) + \
    np.dot(ket_01, bra_01) + \
    np.dot(ket_10, bra_10) + \
    np.dot(ket_11, bra_11)

# TODO: Deduplicate the caculations of h_onedress and h_twodress

def hamiltonian_mwphasecontrol (phi, static_parameters, t,\
    jacobian=True, hessian=False):
    """
    Computes the Hamiltonian matrix and its gradient with respect to
    control variables for phase control

    Parameters
    ----------
    phi: ndarray <float> (1, )
    Phase of the microwave

    static_parameters: ndarray <float> (8, )
    Static parameters of the Hamiltonian in the following order
    [Delta_uv_a, Delta_uv_b, Delta_mw_a, Delta_mw_b,
     Omega_uv_a, Omega_uv_b, Omega_mw_a, Omega_mw_b]

    t: float
    time

    jacobian: boolean
    Whether to calculate the gradient of the Hamiltonian with respect to phase

    hessian: boolean
    Whether to calculate the gradient of the Hamiltonian with respect to phase

    Returns
    -------
    h:
    Hamiltonian matrix

    grad_h:
    Gradient of the Hamiltonian matrixs
    """

    DeltaUV_a = static_parameters[0]
    DeltaUV_b = static_parameters[1]
    DeltaMW_a = static_parameters[2]
    DeltaMW_b = static_parameters[3]
    OmegaUV_a = static_parameters[4]
    OmegaUV_b = static_parameters[5]
    OmegaMW_a = static_parameters[6]
    OmegaMW_b = static_parameters[7]

    h_onedress = -(DeltaUV_a + DeltaMW_a) * np.outer(ket_r0, bra_r0) \
        - (DeltaUV_b + DeltaMW_b) * np.outer(ket_0r, bra_0r) \
        - DeltaMW_a * np.outer(ket_10, bra_10) \
        - DeltaMW_b * np.outer(ket_01, bra_01) \
        + OmegaUV_a / 2 * (np.outer(ket_r0, bra_10) + np.outer(ket_10, bra_r0)) \
        + OmegaUV_b/ 2 * (np.outer(ket_0r, bra_01) + np.outer(ket_01, bra_0r))

    h_twodress = -(DeltaUV_a + DeltaMW_a + DeltaMW_b) * np.outer(ket_r1, bra_r1) \
        -(DeltaUV_b + DeltaMW_a + DeltaMW_b) * np.outer(ket_1r, bra_1r) \
        -(DeltaMW_a + DeltaMW_b) * np.outer(ket_11, bra_11) \
        + OmegaUV_a / 2 * (np.outer(ket_r1, bra_11) + np.outer(ket_11, bra_r1)) \
        + OmegaUV_b / 2 * (np.outer(ket_1r, bra_11) + np.outer(ket_11, bra_1r))

    h_mw = OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(ket_0r, bra_1r) \
                           + exp(+1j*phi) * np.outer(ket_1r, bra_0r))\
          + OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(ket_r0, bra_r1) \
                            + exp(+1j*phi) * np.outer(ket_r1, bra_r0))\
          + OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(ket_01, bra_11)\
                            + exp(+1j*phi) * np.outer(ket_11, bra_01))\
          + OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(ket_10, bra_11)\
                            + exp(+1j*phi) * np.outer(ket_11, bra_10))\
          + OmegaMW_a / 2 * (exp(-1j*phi) * np.outer(ket_00, bra_10)\
                            + exp(+1j*phi) * np.outer(ket_10, bra_00))\
          + OmegaMW_b / 2 * (exp(-1j*phi) * np.outer(ket_00, bra_01)\
                            + exp(+1j*phi) * np.outer(ket_01, bra_00))

    grad_h_mw = 1j * OmegaMW_a / 2 * (-exp(-1j*phi) * np.outer(ket_0r, bra_1r) \
                           + exp(+1j*phi) * np.outer(ket_1r, bra_0r))\
          + 1j * OmegaMW_b / 2 * (-exp(-1j*phi) * np.outer(ket_r0, bra_r1) \
                            + exp(+1j*phi) * np.outer(ket_r1, bra_r0))\
          + 1j * OmegaMW_a / 2 * (-exp(-1j*phi) * np.outer(ket_01, bra_11)\
                            + exp(+1j*phi) * np.outer(ket_11, bra_01))\
          + 1j * OmegaMW_b / 2 * (-exp(-1j*phi) * np.outer(ket_10, bra_11)\
                            + exp(+1j*phi) * np.outer(ket_11, bra_10))\
          + 1j * OmegaMW_a / 2 * (-exp(-1j*phi) * np.outer(ket_00, bra_10)\
                            + exp(+1j*phi) * np.outer(ket_10, bra_00))\
          + 1j * OmegaMW_b / 2 * (-exp(-1j*phi) * np.outer(ket_00, bra_01)\
                            + exp(+1j*phi) * np.outer(ket_01, bra_00))

    h_total = h_onedress + h_twodress + h_mw

    if jacobian:
        return h_total, grad_h_mw
    else:
        return h_total

def hamiltonian_mwdetuningcontrol (detuning, static_parameters, t,\
    jacobian=True, hessian=False):
    """
    Computes the Hamiltonian matrix and its gradient with respect to
    control variables for phase control

    Parameters
    ----------
    detuning: ndarray <float> (1, )
    Detuning of the microwave measured with respect to the baseline microwave
    detuning

    static_parameters: ndarray <float> (8, )
    Static parameters of the Hamiltonian in the following order
    [Delta_uv_a, Delta_uv_b, Delta_mw_a, Delta_mw_b,
     Omega_uv_a, Omega_uv_b, Omega_mw_a, Omega_mw_b]

    Returns
    -------
    h:
    Hamiltonian matrix

    grad_h:
    Gradient of the Hamiltonian matrixs
    """
    phi = detuning * t

    h_total, grad_h_mw = \
        hamiltonian_mwphasecontrol(phi, static_parameters, t, True, False)

    grad_h_mw = grad_h_mw * t

    if jacobian:
        return h_total, grad_h_mw
    else:
        return h_total
