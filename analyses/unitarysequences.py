"""Legacy unitary-sequence analysis helpers for two-qubit control.

This module keeps the original analysis logic used to compare ideal and
uncertain dressing sequences.
"""

import math
import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from scipy.linalg import expm
from numpy import sin, cos, exp, sqrt, pi, sign

import fidelity
import spinoperators

################################################################################
class RydbergAdiabaticPhasesTwoQubits:
    """
    Represents the phases accumulated during adiabatic Rydberg dressing
    and undressing
    """

    def __init__ (self, Ω_uv, Δ_uv, thetatwist=pi, vdd=None):
        self.Ω_uv = Ω_uv
        self.Δ_uv = Δ_uv
        self.thetatwist = thetatwist

        # On resonance, assume the blue detuning branch
        if Δ_uv == 0:
            self.Δ_uv_sign = +1
        else:
            self.Δ_uv_sign = sign(Δ_uv)

        if vdd == None:
            self.vdd = math.inf

    def calc_els1 (self):
        """
        Calculates the light shift due to one atom being dressed independently
        """
        if not hasattr(self, "els1"):
            self.els1 = - self.Δ_uv/2 + \
                self.Δ_uv_sign/2 * sqrt(self.Ω_uv**2 + self.Δ_uv**2)

        return self.els1

    def calc_els2 (self):
        """
        Calculates the light shift due to both atoms being dressed together
        """

        if not hasattr(self, "els2"):
            self.els2 = - self.Δ_uv/2 + \
                self.Δ_uv_sign/2 * sqrt(2*self.Ω_uv**2 + self.Δ_uv**2)

        return self.els2

    def calc_kappa (self):
        """pinoperator.calc_zrotatetwist()
        Calculates the entangling strength which is the difference of the
        lights shift corresponding to atoms being dressed together
        and twice the light shift of the atoms being dressed independently
        """

        if not hasattr(self, "kappa"):
            els1 = self.calc_els1()
            els2 = self.calc_els2()
            self.kappa = els2 - 2*els1

        return self.kappa

    def calc_phases (self):
        """
        Calculates the phases accumulated during adiabatic dressing and
        undressing.

        Assumes that the adiabatic dressing and undressing is designed such
        that the phase accumulated due to `kappa` is pi
        """

        if not hasattr(self, "thetarotate"):
            els1 = self.calc_els1()
            kappa = self.calc_kappa()

            self.timeduration = self.thetatwist / kappa
            self.thetarotate = self.timeduration * (els1 + kappa/2)

        return self.thetatwist, self.thetarotate

################################################################################
################################################################################
class TwoQubitSpin:
    """
    Represents the spin of two qubits
    """

    def __init__ (self):
        spin = spinoperators.TwoQubitSpin()
        self.sx = spin.sx
        self.sy = spin.sy
        self.sz = spin.sz
        self.sxsquared = spin.sxsquared
        self.sysquared = spin.sysquared
        self.szsquared = spin.szsquared
        self.identity = spin.identity

    def get_ndim (self):
        return self.identity.shape[0]

    def get_sx(self):
        return self.sx

    def get_sy(self):
        return self.sy

    def get_sz(self):
        return self.sz

    def get_sxsquared(self):
        return self.sxsquared

    def get_sysquared(self):
        return self.sysquared

    def get_szsquared(self):
        return self.szsquared

    def get_identity(self):
        return self.identity

    def calc_sequator(self, phi):
        """
        Calculates the spin operator in the direction given by angle `phi` from
        the x axis in the equator
        """
        sphi = cos(phi) * self.sx + sin(phi) * self.sy
        return sphi

    def calc_xysu2(self, theta, phi):
        """
        Calculates a special unitary (2) about an axis in the xy plane making
        angle `phi` from the x axis, by angle `theta`
        """
        u = expm(-1j * theta * (cos(phi)*self.sx + sin(phi)*self.sy))
        return u

    def calc_zrotatetwist (self, thetarotate, thetatwist):
        """
        Calculates a twist and rotate unitary about the z axis with
        twist angle `thetatwist` and rotation angle `thetarotate`
        """
        u = expm(-1j * thetarotate * self.sz \
                - 1j * thetatwist * self.szsquared/2)
        return u
    
    def calc_yrotatetwist (self, thetarotate, thetatwist):
        """
        Calculates a twist and rotate unitary about the z axis with
        twist angle `thetatwist` and rotation angle `thetarotate`
        """
        u = expm(-1j * thetarotate * self.sy \
                - 1j * thetatwist * self.sysquared/2)
        return u


################################################################################
################################################################################
class TwoQubitUnitaryQuantumControl:
    """
    Represents a quantum control problem which is implemented as a sequence
    of unitary transformations. Each unitary transformation is parameterized
    by unitary parameters.
    """
    def __init__ (self, target, nsteps, adiabaticphases, spinoperator):
        self.target = target
        self.nsteps = nsteps
        self.nangles = 2
        self.adiabaticphases = adiabaticphases
        self.spinoperator = spinoperator
        self.ndim = spinoperator.get_ndim()

    def get_nvariables(self):
        return self.nangles

    def get_nsteps(self):
        return self.nsteps

    def calc_unitary_uv (self):
        """
        Calculates the unitary transformation implemented during the 
        ultraviolet dressing and undressing stages
        """
        if not hasattr(self, "unitary_uv"):
            thetatwist, thetarotate = self.adiabaticphases.calc_phases()
            self.unitary_uv = \
                self.spinoperator.calc_zrotatetwist(thetarotate, thetatwist)

        return self.unitary_uv

    def set_unitary_uv (self, unitary_uv):
        """
        Sets the unitary transformation implemented uring the ultraviolet
        dressing and undressing stages. This is useful for problems involving
        uncertain parameters and robustness to imperfections
        """
        self.unitary_uv = unitary_uv

    def calc_unitary (self, angles):
        """
        Calculates the unitary transformation implemented by a series of
        special unitary (2) rotations about axes in the equator interleaved
        by rotation and twist unitaries. The rotations are characterized by
        angles `theta` which represents the angle of rotation and `phi` which
        represents the angle made by the axis of rotation from the x axis in
        the xy plane

        The sequence of `theta` is expected in `angles[:, 0]` and the sequence
        of `phi` is expected in `angles[:, 1]`
        """

        angles = np.reshape(angles, (self.nangles, self.nsteps))
        theta = angles[0, :]
        phi = angles[1, :]

        unitary_uv = self.calc_unitary_uv()

        su2_steps = np.zeros((self.ndim, self.ndim, self.nsteps), dtype=complex)
        dsu2_steps_dtheta = np.zeros((self.ndim, self.ndim, self.nsteps), dtype=complex)
        dsu2_steps_dphi = np.zeros((self.ndim, self.ndim, self.nsteps), dtype=complex)

        # Calculate the unitary steps
        for s in range(self.nsteps):
            su2_steps[:, :, s] = self.spinoperator.calc_xysu2(theta[s], phi[s])

        # Calculare the derivatives of the unitary steps
        sz = self.spinoperator.get_sz()
        for s in range(self.nsteps):
            sphi = self.spinoperator.calc_sequator(phi[s])

            dsu2_steps_dtheta[:, :, s] = \
                -1j * sphi * su2_steps[:, :, s]

            dsu2_steps_dphi[:, :, s] = \
                - sz * su2_steps[:, :, s] * sz

        # Calculate the cumulate unitary transformations
        unitary = np.eye(self.ndim, self.ndim, dtype=complex)
        
        #dunitary_dtheta = np.zeros((self.ndim, self.ndim, self.nsteps), dtype=complex)
        #dunitary_dphi = np.zeros((self.ndim, self.ndim, self.nsteps), dtype=complex)

        dunitary_dtheta = \
            np.stack([np.eye(self.ndim, dtype=complex)] * self.nsteps, axis=2)

        dunitary_dphi = \
            np.stack([np.eye(self.ndim, dtype=complex)] * self.nsteps, axis=2)


        for s in range(self.nsteps-1):
            unitary = \
                np.dot(su2_steps[:, :, s], unitary)

            if hasattr(self, "unitary_uv"):
                unitary = \
                    np.dot(self.unitary_uv, unitary)

            # Multiply the unitary steps that occur before the step which is
            # being considered for the derivative
            for ss in range(self.nsteps-1):
                dunitary_dtheta[:, :, s] = \
                    np.dot(su2_steps[:, :, ss], dunitary_dtheta[:, :, s])

                dunitary_dphi[:, :, s] = \
                    np.dot(su2_steps[:, :, ss], dunitary_dphi[:, :, s])

                if hasattr(self, "unitary_uv"):
                    dunitary_dtheta[:, :, s] = \
                        np.dot(unitary_uv, dunitary_dtheta[:, :, s])

                    dunitary_dphi[:, :, s] = \
                        np.dot(unitary_uv, dunitary_dphi[:, :, s])

            # Multiply the derivative of the unitary for the step which is being
            # considered for the derivative
            dunitary_dtheta[:, :, s] = \
                np.dot(dsu2_steps_dtheta[:, :, s], dunitary_dtheta[:, :, s])

            dunitary_dphi[:, :, s] = \
                np.dot(dsu2_steps_dphi[:, :, s], dunitary_dphi[:, :, s])

            if hasattr(self, "unitary_uv"):
                dunitary_dtheta[:, :, s] = \
                    np.dot(unitary_uv, dunitary_dtheta[:, :, s])

                dunitary_dphi[:, :, s] = \
                    np.dot(unitary_uv, dunitary_dphi[:, :, s])

            # Multiply the unitary steps that occur after the step which is
            # being considered for the derivative
            for ss in range(s+1, self.nsteps):
                dunitary_dtheta[:, :, s] = \
                    np.dot(su2_steps[:, :, ss], dunitary_dtheta[:, :, s])
                dunitary_dphi[:, :, s] = \
                    np.dot(su2_steps[:, :, ss], dunitary_dphi[:, :, s])

                if hasattr(self, "unitary_uv"):
                    dunitary_dtheta[:, :, s] = \
                        np.dot(unitary_uv, dunitary_dtheta[:, :, s])

                    dunitary_dphi[:, :, s] = \
                        np.dot(unitary_uv, dunitary_dphi[:, :, s])

        # The final step
        unitary = \
            np.dot(su2_steps[:, :, self.nsteps-1], unitary)

        # Multiply the unitary steps that occur before the step which is
        # being considered for the derivative
        for ss in range(self.nsteps-1):
            dunitary_dtheta[:, :, self.nsteps-1] = \
                np.dot(su2_steps[:, :, ss], dunitary_dtheta[:, :, self.nsteps-1])

            dunitary_dphi[:, :, self.nsteps-1] = \
                np.dot(su2_steps[:, :, ss], dunitary_dphi[:, :, self.nsteps-1])

        # Multiply the derivative of the unitary for the step which is being
        # considered for the derivative
        dunitary_dtheta[:, :, self.nsteps-1] = \
            np.dot(dsu2_steps_dtheta[:, :, self.nsteps-1], \
                   dunitary_dtheta[:, :, self.nsteps-1])

        dunitary_dphi[:, :, self.nsteps-1] = \
            np.dot(dsu2_steps_dphi[:, :, self.nsteps-1], \
                   dunitary_dphi[:, :, self.nsteps-1])


        return unitary, dunitary_dtheta, dunitary_dphi

    def calc_infidelity(self, angles, args=None):
        """
        Calculates the infidelity of implementing a target unitary
        """

        unitary, dunitary_dtheta, dunitary_dphi = \
            self.calc_unitary(angles)

        denominator = np.min([\
                np.linalg.matrix_rank(unitary), \
                np.linalg.matrix_rank(self.target)])

        goal_value = np.trace(np.dot(unitary, fidelity.dagger(self.target))) \
                    / denominator
    
        infidelity_value = 1 - np.abs(goal_value)**2

        dgoal_dtheta = np.zeros(self.nsteps, dtype=complex)
        dgoal_dphi = np.zeros(self.nsteps, dtype=complex)

        for s in range(self.nsteps):
            dgoal_dtheta[s] = (np.trace(np.dot(dunitary_dtheta[:, :, s],\
                fidelity.dagger(self.target)))) / denominator

            dgoal_dphi[s] = (np.trace(np.dot(dunitary_dphi[:, :, s],\
                fidelity.dagger(self.target)))) / denominator

        dinfidelity_values = np.empty((self.nangles, self.nsteps))
        dinfidelity_values[0, :] = \
            - 2*np.real(dgoal_dtheta * np.conjugate(goal_value))
            
        dinfidelity_values[1, :] = \
            - 2*np.real(dgoal_dphi * np.conjugate(goal_value))

        dinfidelity_values_flattened = \
            np.reshape(dinfidelity_values, self.nsteps * self.nangles)

        return infidelity_value, dinfidelity_values_flattened
################################################################################
################################################################################
class ControlSequenceSearch:
    """
    Represents a search for a control sequence to implement a
    """

    def __init__ (self, system, gtol=1e-4, maxiter=1024, variable_bounds=None):
        self.system = system
        self.nsteps = system.get_nsteps()
        self.nvariables = system.get_nvariables()

        self.variable_bounds = variable_bounds

        self.gtol = gtol
        self.maxiter = maxiter
        self.jac=False#True

    def calc_optsequence (self, variables_initial, debug=True):
        """
        Searches for the optimizing waveforms by using bfgs / lbfgs
        """

        cost_function = self.system.calc_infidelity

        if self.variable_bounds is not None:
            result = scipy.optimize.minimize(\
                  fun=cost_function, x0=variables_initial, jac=self.jac, method='L-BFGS-B', \
                  bounds=self.variable_bounds, \
                  options={'gtol': self.gtol, 'maxiter': self.maxiter,}, \
                  args=None)

        else:
            result = scipy.optimize.minimize(\
                 fun=cost_function, x0=variables_initial, jac=self.jac, method='BFGS', \
                 options={'gtol': self.gtol, 'maxiter': self.maxiter,}, \
                 args=None)


        variables_optimized = result.x
        infidelity_min = result.fun
        dinfidelity_min = result.jac
        status = result.status
        niterations = result.nit

        if debug:
            print('# dinfidelity_min = %s\n # infidelity_min = %g\n # Niterations = %d' \
                   % (dinfidelity_min, infidelity_min, niterations))
            print(result)

        results = {\
                   'variables_initial': variables_initial, \
                   'variables_optimized' : variables_optimized, \
                   'infidelity_min' : infidelity_min, \
                   'dinfidelity_min' : dinfidelity_min, \
                   'niterations' : niterations, \
        }

        self.minimize_result = results
        self.result = result

        return variables_optimized, infidelity_min
################################################################################
################################################################################
class UncertainTwoQubitUnitaryQuantumControl:
    """
    Represents a quantum control problem which is implemented as a sequence
    of unitary transformations. Each unitary transformation is parameterized
    by unitary parameters.
    """

    def __init__ (self, target, nsteps, adiabaticphases, spinoperator, \
            distribwidth, nlandmarks=7):
        self.target = target
        self.nsteps = nsteps
        self.nangles = 2
        self.adiabaticphases = adiabaticphases
        self.spinoperator = spinoperator

        self.distribwidth = distribwidth
        self.nlandmarks = nlandmarks
        self.ndim = spinoperator.get_ndim()

    def get_nvariables(self):
        return self.nangles

    def get_nsteps(self):
        return self.nsteps

    def calc_landmarks (self):
        """
        Calculates the landmark points and their weights
        """

        if self.distribwidth <= 0:
            raise ValueError("distribwidth must be positive")

        cache_key = (self.nlandmarks, self.distribwidth)
        if getattr(self, "_landmark_cache_key", None) != cache_key:
            half_span = int(self.nlandmarks / 2) * self.distribwidth
            self.landmarks = np.linspace(-half_span, half_span, self.nlandmarks)
            self.weights = scipy.stats.norm.pdf(self.landmarks, scale=self.distribwidth)
            self.weights /= np.sum(self.weights)
            self._landmark_cache_key = cache_key

        return self.landmarks, self.weights
    
    def calc_systems (self):
        """
        Prepares the systems which represent different members of the ensemble
        which needs to be addressed
        """
        landmarks, weights = self.calc_landmarks()
        
        cache_key = (self.nlandmarks, self.distribwidth)
        if getattr(self, "_systems_cache_key", None) != cache_key:
            systems = []
    
            for l in range(self.nlandmarks):
                qc = TwoQubitUnitaryQuantumControl(\
                    self.target, self.nsteps, self.adiabaticphases, \
                    self.spinoperator)
    
                thetatwist, thetarotate = self.adiabaticphases.calc_phases()
                unitary_uv = self.spinoperator.calc_zrotatetwist(\
                        thetarotate * (1+self.landmarks[l]), \
                        thetatwist * (1+self.landmarks[l]))
                qc.set_unitary_uv(unitary_uv)
                
                systems.append(qc)
                
            self.systems = systems
            self._systems_cache_key = cache_key
            
        return self.systems



    def calc_infidelity (self, angles, args=None):
        """
        Calculates a weighted sum of infidelity over each landmark points
        """
        
        systems = self.calc_systems()
        
        infidelity_values = np.ones(self.nlandmarks)
        
        for l in range(self.nlandmarks):
            qc = systems[l]
            infidelity_value, _ = qc.calc_infidelity(angles)

            infidelity_values[l] = infidelity_value

        infidelity_mean = np.sum(self.weights * infidelity_values) 

        return infidelity_mean
