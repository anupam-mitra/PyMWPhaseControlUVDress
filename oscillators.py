import numpy as np

from numpy import exp, sqrt

from fidelity import dagger


class HarmonicOscillator:
    def __init__(self, nmax, omega, hbar):
        self.nmax = nmax
        self.omega = omega
        self.hbar = hbar

    def calc_identity(self):
        self.identity = np.eye(self.nmax+1)

    def get_identity(self):
        if not hasattr(self, "identity"):
            self.calc_identity()
        return self.identity

    def calc_a(self):
        self.a = np.diag([sqrt(n) for n in np.arange(1, self.nmax+1)], 1)

    def get_a(self):
        if not hasattr(self, "a"):
            self.calc_a()
        return self.a

    def calc_adag(self):
        self.adag = dagger(self.get_a())

    def get_adag(self):
        if not hasattr(self, "adag"):
            self.calc_adag()
        return self.adag

    def calc_numop(self):
        self.n = np.dot(self.get_adag(), self.get_a())

    def get_numop(self):
        if not hasattr(self, "n"):
            self.calc_numop()
        return self.n

    def calc_ham(self):
        self.ham = (self.get_numop() + 1/2 * self.get_identity()) * self.hbar * self.omega

    def get_ham(self):
        if not hasattr(self, "ham"):
            self.calc_ham()
        return self.ham

    def get_quadop(self, phi):
        return (self.get_a() * exp(1j*phi) + self.get_adag() * exp(-1j*phi)) / sqrt(2)


class DampedHarmonicOscillator(HarmonicOscillator):
    def __init__(self, nmax, omega, hbar, gamma):
        HarmonicOscillator.__init__(self, nmax, omega, hbar)
        self.gamma = gamma

    def calc_hameff(self):
        self.hamdecay = np.dot(self.get_numop(), self.get_numop()) * (-1j * self.gamma)
        self.hameff = self.get_ham() + self.hamdecay

    def get_hameff(self):
        if not hasattr(self, "hameff"):
            self.calc_hameff()
        return self.hameff
