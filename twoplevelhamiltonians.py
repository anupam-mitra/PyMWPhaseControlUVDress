#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  twoplevelhamiltonians.py
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
#
from __future__ import division

import numpy as np
from numpy import exp, cos, sin

ket_g = np.asarray([0, 1], dtype=complex)
ket_e = np.asarray([1, 0], dtype=complex)

bra_g = np.transpose(np.conjugate(ket_g))
bra_e = np.transpose(np.conjugate(ket_e))

sigma_x = np.outer(ket_e, bra_g) + np.outer(ket_g, bra_e)
sigma_y = 1j * np.outer(ket_e, bra_g) - 1j * np.outer(ket_g, bra_e)
sigma_z = np.outer(ket_e, bra_e) - np.outer(ket_g, bra_g)

def hamiltonian_phase_control (phase, detuning, rabifrequency, \
    jacobian=True, hessian=False):
    """
    Computes the Hamiltonian for phase control, in the
    rotating frame
    """

    h = rabifrequency/2 * \
        (cos(phase) * sigma_x + sin(phase) * sigma_y) \
        - detuning/2 * sigma_z

    dh_dphase = rabifrequency/2 * \
            (-sin(phase) * sigma_x + cos(phase) * sigma_y)
    
    d2h_dphase2 = -rabifrequency/2 * \
            (cos(phase) * sigma_x + sin(phase) * sigma_y)

    if not jacobian:
        return h
    elif not hessian:
        return h, dh_dphase
    else:
        return h, dh_dphase, d2h_dphase2

def hamiltonian_detuning_control (detuning, t, rabifrequency, \
        jacobian=True, hessian=False):
    """
    Computes the Hamiltonian for detuning control, in the
    rotating frame at the Bohr frequency
    """

    h = rabifrequency/2 * \
        (cos(detuning * t) * sigma_x \
          + sin(detuning * t) * sigma_y)

    dh_ddetuning = rabifrequency/2 * t * \
        (- sin(detuning * t) * sigma_x \
          + cos(detuning * t) * sigma_y)

    d2h_ddetuning = -rabifrequency/2 * t*t *\
        (cos(detuning * t) * sigma_x \
          + sin(detuning * t) * sigma_y)

    if not jacobian:
        return h
    elif not hessian:
        return h, dh_ddetuning
    else:
        return h, dh_dphase, d2h_ddetuning

def hamiltonian_rabifrequency_quadratures_control \
    (detuning, rabifrequency_c, rabi_frequency_s, \
     jacobian=True, hessian=False):
    """
    Computes the Hamiltonian for quadratures
    of the rabi frequency in the rotating frame
    """

    h = rabifrequency_c/2 * sigma_x \
        + rabifrequency_s/2 *  sigma_y \
        - detuning/2 * sigma_z

    dh = np.empty(h.shape + (2, ))
    d2h = np.empty(h.shape + (2, 2))
    
    dh[:, :, 0] = 1/2 * sigma_x
    dh[:, :, 1] = 1/2 * sigma_y
    
    d2h[:, :, 0, 0] = np.zeros(sigma_x.shape)
    d2h[:, :, 0, 1] = np.zeros(sigma_y.shape)
    d2h[:, :, 1, 0] = np.zeros(sigma_x.shape)
    d2h[:, :, 1, 1] = np.zeros(sigma_y.shape)

    if not jacobian:
        return h
    elif not hessian:
        return h, dh
    else:
        return h, dh, d2h

def hamiltonian_phase_control_old (phase, h_params):
    detuning = h_params['Delta']
    rabifrequency = h_params['Omega']

    h = hamiltonian_phase_control(phase, detuning, rabifrequency, False, False)

    return h

def hamiltonian_phase_control_grad_old (phase, h_params):
    detuning = h_params['Delta']
    rabifrequency = h_params['Omega']

    _, dh = hamiltonian_phase_control(phase, detuning, rabifrequency, True, False)

    return dh
