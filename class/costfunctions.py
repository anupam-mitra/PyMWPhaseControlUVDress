from __future__ import division

import numpy as np
from numpy import exp

from ..utilities import dagger


class Infidelity:
    """
    Represents a cost function based on infidelity of implementing a particular
    time evolution, either completely or partially

    Parameters
    ----------
    target:
    Target time evolution operator

    hamiltonian_function:
    Python function which calculates the Hamiltonians
    """

    def __init__ (self, target):
        self.target = target
        self.target_dagger = dagger(target)
        self.rank_target = np.linalg.matrix_rank(target)


    def infidelity_from_unitary (self, u, u_gradient):
        """
        Calculate the infidelity and its gradient given the time evolution
        operator and its gradient
        """
        nsteps = u_gradient.shape[-2]
        ncontrols = u_gradient.shape[-1]

        target = self.target
        target_dagger = self.target_dagger
        rank_target = self.rank_target

        goal = np.trace(np.dot(target_dagger, u))
        infidelity_value = (np.abs(goal) / rank_target)**2

        goal_gradient = np.zeros(nsteps, ncontrols)
        infidelity_gradient = np.zeros(nsteps, ncontrols)

        for c in range(ncontrols):
            for s in range(nsteps):
                goal_gradient[s, c] = np.trace(np.dot(target_dagger, \
                                        u_gradient[:, :, s, c]))

                infidelity_gradient = - 2 * np.real(\
                    goal_gradient[s, c] * np.conjugate(goal))

        return infidelity_value, infidelity_gradient


    def calc_hamiltonians (self, control):
        pass


class EnsembleInfidelity:
    """
    Represents a cost function for an infidelity evaluated for an ensemble of
    landmarks points approximating a distribution

    """
    def __init__ (self):
        pass


    def pseudocode ():

        # Loop over landmarks

        # Calculate the time evolution operator and its gradient
        # for each landmark

        # Calculate the infidelity at each landmark point

        # Average the infidelity over all landmark points
        pass
