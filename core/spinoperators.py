import numpy as np

from numpy import cos, sin, sqrt
from scipy.linalg import expm

from core.fidelity import dagger


def angular_momentum_operators(s):
    """
    Generates angular momentum operators for spin s.
    """
    sz = np.diag([m for m in np.arange(s, -s-1, -1)])
    sminus = np.diag([sqrt(s*(s+1) - m*(m-1)) for m in np.arange(s, -s, -1)], 1)
    splus = dagger(sminus)
    sx = (splus + sminus) / 2
    sy = (splus - sminus) * (-1j) / 2
    return sx, sy, sz


def two_qubit_spin_operators():
    sigmax, sigmay, sigmaz = (2*s for s in angular_momentum_operators(1/2))
    identity2 = np.eye(2)

    sx = (np.kron(identity2, sigmax) + np.kron(sigmax, identity2)) / 2
    sy = (np.kron(identity2, sigmay) + np.kron(sigmay, identity2)) / 2
    sz = (np.kron(identity2, sigmaz) + np.kron(sigmaz, identity2)) / 2

    return sx, sy, sz


class TwoQubitSpin:
    def __init__(self):
        self.sx, self.sy, self.sz = two_qubit_spin_operators()
        self.sxsquared = np.dot(self.sx, self.sx)
        self.sysquared = np.dot(self.sy, self.sy)
        self.szsquared = np.dot(self.sz, self.sz)
        self.identity = np.eye(4)

    def get_ndim(self):
        return 4

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
        return cos(phi) * self.sx + sin(phi) * self.sy

    def calc_xysu2(self, theta, phi):
        return expm(-1j * theta * self.calc_sequator(phi))

    def calc_zrotatetwist(self, thetarotate, thetatwist):
        return expm(-1j * thetarotate * self.sz - 1j * thetatwist * self.szsquared/2)

    def calc_yrotatetwist(self, thetarotate, thetatwist):
        return expm(-1j * thetarotate * self.sy - 1j * thetatwist * self.sysquared/2)
