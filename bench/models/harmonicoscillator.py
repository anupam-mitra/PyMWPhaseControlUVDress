import numpy as np

from numpy import sqrt

dagger = lambda u: np.conjugate(np.transpose(u))

################################################################################
class HarmonicOscillator:
    """
    Represents a quantum harmonic oscillator
    """

    def __init__ (self, nmax, omega, hbar):
        print("Printing from HarmonicOscillator constructor")
        self.nmax = nmax
        self.omega = omega
        self.hbar = hbar


    def calc_identity (self):
        """
        Computes the identity operator and stores it as an attribute
        """
        self.identity = np.eye(self.nmax+1)

    def get_identity (self):
        """
        Returns the identity operator
        """
        if not hasattr(self, "identity"):
            self.calc_identity()

        return self.identity

    def calc_a (self):
        """
        Computes the annhilation operator a and stores it as an attribute
        """
        self.a = np.diag([sqrt(n) for n in np.arange(self.nmax)], 1)

    def get_a (self):
        """
        Returns the annihilation operator a
        """
        if not hasattr(self, "a"):
            self.ladderop()

        return self.a

    def calc_adag (self):
        """
        Computes the creation operator a and stores it as an attribute
        """

        if not hasattr(self, "a"):
            self.calc_a()

        self.adag = dagger(self.a)

    def get_adag (self):
        """
        Returns the creation operator adag
        """
        if not hasattr(self, "adag"):
            self.calc_adag()

        return self.adag

    def calc_numop (self):
        """
        Computes the number operator and stores it as an attribute
        """
        if not (hasattr(self, "a") or hasattr(self, "adag")):
            self.calc_a()
            self.calc_adag()

        self.n = np.dot(self.adag, self.a)

    def get_numop (self):
        """
        Returns the number operator
        """
        if hasattr(self, "n"):
            self.calc_numop()

        return self.n

    def calc_ham (self):
        """
        Computes the Hamiltonian and stores it as an attribute
        """
        if not hasattr(self, "n"):
            self.calc_numop()

        if not hasattr(self, "identity"):
            self.calc_identity()

        self.ham = (self.n + 1/2 * self.identity) * self.hbar * self.omega

    def get_ham (self):
        """
        Returns the Hamiltonian ham
        """
        if not hasattr(self, "ham"):
            self.calc_ham()

        return self.ham

    def get_quadop (self, phi):
        """
        Computes and returns a quadrature operator with phase angle phi
        """

        if not (hasattr(self, "a") or hasattr(self, "adag")):
            self.calc_a()
            self.calc_dag()

        xphi = (self.a * exp(1j*phi) + self.adag * exp(-1j*phi)) / sqrt(2)

        return xphi
################################################################################
################################################################################
class DampedHarmonicOscillator (HarmonicOscillator):
    """
    Class representing a damped quantum harmonic oscillator
    """
    def __init__(self, nmax, omega, hbar, gamma):
        print("Printing from DampedHarmonicOscillator constructor")
        HarmonicOscillator.__init__(self, nmax, omega, hbar)
        self.gamma = gamma

    def calc_hameff(self):
        """
        Computes the effective Hamiltonian and stores it as an attribute
        """
        if not hasattr(self, "ham"):
            self.calc_ham()

        self.hamdecay = np.dot(self.num, self.num) * (-im * gamma)

        self.hameff = self.ham + self.hamdecay

    def get_hameff (self):
        """
        Returns the effective Hamiltonian with the decay term added to it
        """
        if not hasattr(self, "hameff"):
            self.calc_hameff()

        return self.hameff
################################################################################
