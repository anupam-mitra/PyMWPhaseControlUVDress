from __future__ import division

import numpy as np
from numpy import exp

import propagators

class TDSESolver:
    """
    Solves the time dependent Schrodinger equation. This is a base class which
	represents a solver.

    """
    def __init__ (self, system):
        self.system = system
        self.ndim = ndim
        self.basis = basis
        
        self.tinitial = tinitial
        self.tfinal = tfinal
        
        self.initialcondition = initialcondition


    def calc_rhs (self, t, u):
        """
        Calculates the right side of the time dependent Schrodinger equation
        """
        
        h = self.system.calc_hamiltonian()

class PiecewisePropagator (TDSESolver):
    """
    Solves the time dependent Schrodinger equation using the exponentiation of
    a piece wise constant Hamiltonian, either by design (for example when doing
    piece wise constant "pulses"), or by approximation.

	This is a base class which represents piecewise constant solver

    Parameters
    ----------
    nsteps: Number of time steps to for control

    tstep: Temporal duration of each step

    """
    def __init__ (self, nsteps, tstep, hamiltonians, hamiltonian_gradient):
        self.nsteps = nsteps
		self.tstep = tstep
		self.tcontrol = tstep * nsteps

		self.hamiltonians = hamiltonians
		self.hamiltonian_gradient = hamiltonian_gradient

		self.ndim = hamiltonians.shape[0]

	def calc_propagator (self):
		"""
		Calculates the time evolution operator and its gradient with respect
		to control variables
		"""

		if hasttr(self, 'propagator') and hasattr(self, 'propagator_gradient')
			u = self.propagator
			u_gradient = self.propagator_gradient

			return u, u_gradient

		nsteps = self.nsteps
		tstep = self.tstep
		ndim = self.ndim

		hamiltonians = self.hamiltonians
		hamiltonian_gradient = self.hamiltonian_gradient

		u = np.identity(ndim, dtype=complex)
		u_gradient = np.empty(ndim, ndim, nsteps, ncontrols)

		u_steps = np.empty(ndim, ndim, nsteps)

		for s in range(nsteps):
			h = hamiltonians[:, :, s]
			u_steps[s] = expm(-1j * tstep * h)

			u  = np.dot(u_steps[s], u)

		for c in range(ncontrols):
			for s in range(nsteps):
				u_gradient[:, :, s, c] = np.identity(ndim, dtype=complex)

				for st in range(s):
					u_gradient[:, :, s, c] = \
						np.dot(u_steps[st], u_gradient[:, :, s, c])

				u_step_derivative = propagator_step_derivative(\
					hamiltonians[:, :, s], hamiltonian_gradient[:, :, s, c])

				u_gradient[:, :, s, c] = np.dot(u_step_derivative, \
								u_gradient[:, :, s, c)

				for st in range(s+1, nsteps):
					u_gradient[:, :, s, c] = \
						np.dot(u_steps[st], u_gradient[:, :, s, c])

		self.propagator = u
		self.propagator_gradient = u_gradient

		return u, u_gradient


class RungeKuttaPropagator (TDSESolver):
    """
    Solves the time dependent Schrodinger equation using the Runge Kutta method
    """
    pass

class PiecewisePropagatorSaveHistory (TDSESolver):
    """
    Solves the time dependent Schrodinger equation using the exponentiation of
    a piece wise constant Hamiltonian, either by design (for example when doing
    piece wise constant "pulses"), or by approximation. Also saves the
    propagator at intermediate steps.
    """
    pass
