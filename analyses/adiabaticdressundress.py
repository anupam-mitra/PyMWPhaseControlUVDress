import itertools
import os
import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

from scipy.linalg import expm
from numpy import pi, cos, sin, exp, sqrt, arctan2
dagger = lambda u : np.transpose(np.conjugate(u))

ket_r1 = np.asarray([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_1r = np.asarray([[0], [1], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_r0 = np.asarray([[0], [0], [1], [0], [0], [0], [0], [0]], dtype=complex)
ket_0r = np.asarray([[0], [0], [0], [1], [0], [0], [0], [0]], dtype=complex)
ket_11 = np.asarray([[0], [0], [0], [0], [1], [0], [0], [0]], dtype=complex)
ket_10 = np.asarray([[0], [0], [0], [0], [0], [1], [0], [0]], dtype=complex)
ket_01 = np.asarray([[0], [0], [0], [0], [0], [0], [1], [0]], dtype=complex)
ket_00 = np.asarray([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=complex)

bra_00 = ket_00.conjugate().transpose()
bra_01 = ket_01.conjugate().transpose()
bra_0r = ket_0r.conjugate().transpose()
bra_10 = ket_10.conjugate().transpose()
bra_11 = ket_11.conjugate().transpose()
bra_1r = ket_1r.conjugate().transpose()
bra_r0 = ket_r0.conjugate().transpose()
bra_r1 = ket_r1.conjugate().transpose()

def phase (t):
    phi = 0
    return phi


def hamiltonian (t, adiabatic_dressing, DeltaMW, OmegaMW):
    pf = exp(-1j * phase(t))
    pfc = np.conjugate(pf)

    OmegaR, dOmegaR_dt = adiabatic_dressing.get_Omega(t)
    DeltaR, dDeltaR_dt = adiabatic_dressing.get_Delta(t)


    h = \
        - (DeltaR + 2*DeltaMW) * np.outer(ket_r1, bra_r1) \
        - (DeltaR + 2*DeltaMW) * np.outer(ket_1r, bra_1r) \
        - (DeltaR + DeltaMW) * np.outer(ket_r0, bra_r0) \
        - (DeltaR + DeltaMW) * np.outer(ket_0r, bra_0r) \
        - 2*DeltaMW * np.outer(ket_11, bra_11) \
        - DeltaMW * np.outer(ket_10, bra_10) \
        - DeltaMW * np.outer(ket_01, bra_01) \
        + OmegaR/2 * (np.outer(ket_r1, bra_11) + np.outer(ket_11, bra_r1)) \
        + OmegaR/2 * (np.outer(ket_1r, bra_11) + np.outer(ket_11, bra_1r)) \
        + OmegaR/2 * (np.outer(ket_r0, bra_10) + np.outer(ket_10, bra_r0)) \
        + OmegaR/2 * (np.outer(ket_0r, bra_01) + np.outer(ket_01, bra_0r)) \
        + OmegaMW/2 * (pf * np.outer(ket_11, bra_10) + pfc * np.outer(ket_10, bra_11)) \
        + OmegaMW/2 * (pf * np.outer(ket_11, bra_01) + pfc * np.outer(ket_01, bra_11)) \
        + OmegaMW/2 * (pf * np.outer(ket_10, bra_00) + pfc * np.outer(ket_00, bra_10)) \
        + OmegaMW/2 * (pf * np.outer(ket_01, bra_00) + pfc * np.outer(ket_00, bra_01))

    return h


class RydbergAdiabaticDressUndress:
    """
    Represent an adiabatic sweep of Rabi frequency and detuning
    for a two level atom

    Parameters
    ----------
    Omega_min: Minimum Rabi frequency

    Omega_max: Maximum Rabi frequency

    Delta_min: Minimum detuning

    Delta_max: Maximum detuning

    t_gaussian_width: Width (variance like quantity) of the Gaussian sweep

    t_constant_duration: Duration for which the laser's parameters are constant

    gausswidthslinear: Number of gaussian widths during the linear detuning
    sweep fro Delta_min to Delta_max

    t_mid: Mid point of the adiabatic evolution

    t_dress_begin: Time of beginning the dressing sweep

    t_dress_end: Time of ending the dressing sweep

    t_undress_begin: Time of beginning the undressing sweep

    t_undress_begin: time of ending the undressing sweep
    """
    def __init__ (self, Omega_min, Omega_max, Delta_min, Delta_max, \
                 t_gaussian_width, t_constant_duration, t_mid, \
                 gausswidthslinear=2):
        self.Omega_min = Omega_min
        self.Omega_max = Omega_max
        self.Delta_min = Delta_min
        self.Delta_max = Delta_max

        self.t_gaussian_width = t_gaussian_width
        self.t_constant_duration = t_constant_duration
        self.gausswidthslinear = gausswidthslinear

        self.t_gaussian_duration = gausswidthslinear*t_gaussian_width

        self.t_dress_begin = t_mid - self.t_gaussian_duration - t_constant_duration/2
        self.t_dress_end = t_mid - t_constant_duration/2
        self.t_undress_begin = t_mid + t_constant_duration/2
        self.t_undress_end = t_mid + self.t_gaussian_duration + t_constant_duration/2
        
        self.sign = np.sign(Delta_max)


    def get_Omega (self, t):
        """
        Computes the Rabi frequency at time t

        Parameters
        ----------
        t_width: width of the Gaussian

        t: time

        Returns
        ------
        Omega: Rabi frequency

        dOmega_dt: Derivative of the Rabi frequency with respect
            to time
        """

        if t < self.t_dress_begin or t > self.t_undress_end:
            Omega = self.Omega_min
            dOmega_dt = 0

        elif t < self.t_dress_end:
            t_zeroed = t - self.t_dress_end
            gaussian_factor = exp(-t_zeroed **2 / 2/self.t_gaussian_width**2)

            Omega = self.Omega_min \
            + (self.Omega_max - self.Omega_min) * gaussian_factor

            dOmega_dt = - (self.Omega_max - self.Omega_min) * t_zeroed / (self.t_gaussian_width)**2 \
               * gaussian_factor

        elif t > self.t_undress_begin:
            t_zeroed = t - self.t_undress_begin

            gaussian_factor = exp(-t_zeroed**2 / 2/self.t_gaussian_width**2)

            Omega = self.Omega_min \
            + (self.Omega_max - self.Omega_min) * gaussian_factor

            dOmega_dt = - (self.Omega_max - self.Omega_min) * t_zeroed / (self.t_gaussian_width)**2 \
               * gaussian_factor

        else:
            Omega = self.Omega_max
            dOmega_dt = 0

        return Omega, dOmega_dt

    def get_Delta (self, t):
        """
        Computes the detuning at time t

        Parameters
        ----------
        t: time

        Returns
        -------
        Delta: Detuning

        dDelta_dt: Derivative of the detuning with respect to time
        """

        if t < self.t_dress_begin or t > self.t_undress_end:
            Delta = self.Delta_max
            dDelta_dt = 0

        elif t < self.t_dress_end:
            t_zeroed = t - self.t_dress_begin
            Delta = self.Delta_max \
                + (self.Delta_min - self.Delta_max) / (self.t_dress_end - self.t_dress_begin) * t_zeroed
            dDelta_dt = (self.Delta_min - self.Delta_max) / (self.t_dress_end - self.t_dress_begin)

        elif t > self.t_undress_begin:
            t_zeroed = t - self.t_undress_begin
            Delta = self.Delta_min \
                + (self.Delta_max - self.Delta_min) / (self.t_undress_end - self.t_undress_begin) * t_zeroed
            dDelta_dt = (self.Delta_max - self.Delta_min) / (self.t_undress_end - self.t_undress_begin)
        else:
            Delta = self.Delta_min
            dDelta_dt = 0

        return Delta, dDelta_dt

    def get_kappa (self, t):
        """
        Computes the entangling energy at time t

        Parameters
        ----------
        t: time

        Returns
        -------
        kappa: Entangling energy

        dkappa_dt: Derivative of the entangling energy with respect to time
        """

        Omega, dOmega_dt = self.get_Omega(t)
        Delta, dDelta_dt = self.get_Delta(t)

        kappa = Delta/2 + self.sign * \
            (sqrt(Delta**2 + 2*Omega**2) / 2 - sqrt(Delta**2 + Omega**2))

        return kappa

    def get_admix_angles (self, t):
        """
        Computes the admixing angles at time t
        
        Parameters
        ----------
        t: time
        
        Returns
        -------
        theta_admix_1: admixing angle for each atom being dressed
                independently
                
        
        theta_admix_2: admixing angle for both atoms being dressed
                together
        """
        
        Omega, dOmega_dt = self.get_Omega(t)
        Delta, dDelta_dt = self.get_Delta(t)

        theta_admix_1 = arctan2(Omega, Delta)
        theta_admix_2 = arctan2(sqrt(2) * Omega, Delta)
        
        return theta_admix_1, theta_admix_2

    def get_diabaticity (self, t):
        """
        Computes the diabaticity at time t

        Parameters
        ----------
        t: time

        Returns
        -------
        diabaticity: Diabaticity at time t
        """

        Omega, dOmega_dt = self.get_Omega(t)
        Delta, dDelta_dt = self.get_Delta(t)

        diabaticity = np.abs((Omega * dDelta_dt - Delta * dOmega_dt) / \
                            (Omega**2 + Delta**2)**(3/2))

        return diabaticity


    def get_entangling_phase (self, numpoints=512):
        """
        Computes the entangling phase accumulated by an adiabatic ramp

        Parameters
        ----------
        numpoints: Number of points to use in the time domain to compute
                the integral

        Returns
        -------
        phase_entangle: Entangling phase accumulated

        """

        t = np.linspace(self.t_dress_begin, self.t_undress_end, num=numpoints)

        Omega = np.empty(t.shape)
        dOmega_dt = np.empty(t.shape)
        Delta = np.empty(t.shape)
        dDelta_dt = np.empty(t.shape)

        kappa = np.empty(t.shape)

        for n in range(t.shape[0]):
            Omega[n], dOmega_dt[n] = self.get_Omega(t=t[n])
            Delta[n], dDelta_dt[n] = self.get_Delta(t=t[n])

            kappa[n] = self.get_kappa(t=t[n])

        phase_entangle = scipy.integrate.cumtrapz(kappa, t)[-1]

        return phase_entangle


    def get_integrated_rydberg_population (self, numpoints=512):
        """
        Computes the integrated Rydberg population
        
        Parameters
        ----------
        numpoints: Number of points to use in the time domain to compute
                the integral

        Returns
        -------
        integrated_rydberg_population: Estimate of integrated Rydberg
            population
        """

        t = np.linspace(self.t_dress_begin, self.t_undress_end, num=numpoints)
        
        
        theta_admix_1 = np.empty(t.shape)
        theta_admix_2 = np.empty(t.shape)
        
        for n in range(t.shape[0]):
            theta_admix_1[n], theta_admix_2[n] = self.get_admix_angles(t[n])

        integrated_rydberg_population = \
            2/3*scipy.integrate.cumtrapz(sin(theta_admix_1/2)**2, t) + \
            1/3*scipy.integrate.cumtrapz(sin(theta_admix_2/2)**2, t)
        
        
        return integrated_rydberg_population
        
    def get_decay_estimate (self, numpoints=512):
        """
        Computes the estimate of decay
        
        Parameters
        ----------
        numpoints: Number of points to use in the time domain to compute
                the integral

        Returns
        -------
        prob_decay: Estimate of integrated Rydberg
            population
        """

        integrated_rydberg_population = \
            self.get_integrated_rydberg_population(numpoints)
            
        self.Gamma_r = 1/600

        prob_decay = integrated_rydberg_population * self.Gamma_r
        
        return prob_decay

def hamiltonian_derivative (t, adiabatic_dressing, DeltaMW, OmegaMW):
    OmegaR, dOmegaR_dtwidth = adiabatic_dressing.get_Omega(t)

    dH_dOmega \
        = 1/2 * (np.outer(ket_r1, bra_11) + np.outer(ket_11, bra_r1)) \
        + 1/2 * (np.outer(ket_1r, bra_11) + np.outer(ket_11, bra_1r)) \
        + 1/2 * (np.outer(ket_r0, bra_10) + np.outer(ket_10, bra_r0)) \
        + 1/2 * (np.outer(ket_0r, bra_01) + np.outer(ket_01, bra_0r)) \

    dH_dOmega *= dOmegaR_dtwidth
    return dH_dOmega
