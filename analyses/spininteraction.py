import math
import sys
import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from scipy.linalg import expm
from numpy import sin, cos, exp, sqrt, pi, sign

sys.path.append('../bench/models')

dagger = lambda u: np.transpose(np.conjugate(u))

import angularmomentum



class SpinSystem:
    """
    Represents a system of spin 1/2 with interactions
    along the z direction and a transversefield along
    the x direction
    """

    def __init__ (self, nspins, interactions, transversefield):
        self.nspins = nspins

        self.interactions = interactions
        self.transversefield = transversefield

        self.sigmax, self.sigmay, self.sigmaz = \
            (2*s for s in angularmomentum.angularmomentumop(1/2))

        self.identity2 = np.eye(2)

        self.ndim = 2**nspins

    def calc_hamiltonian (self):
        if not hasattr(self, "hamiltonian"):
            self.hamiltonian = self._construct_hamiltonian()

        return self.hamiltonian

    def _construct_hamiltonian (self):
        htransverse = np.empty((self.ndim, self.ndim))

        for j in range(self.nspins):
            htransverse_term = 1
            for k in range(j)
                htransverse_term = np.kron(htransverseterm, self.identity)

            htransverse_term = np.kron(htransverse_term, self.sigmax)

            for k in range(j+1, self.nspins):
                htransverse_term = np.kron(htransverseterm, self.identity)

            htransverse += htransverse_term
