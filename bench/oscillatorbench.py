import numpy as np
import scipy

from core import oscillators

from numpy import sqrt
from scipy.linalg import expm

def main():
    noscillators = 2
    hbar = 1
    omega = 1
    gamma = 1/8
    kappa = 1/4
    nmax = 3

    systems = [oscillators.DampedHarmonicOscillator(nmax, omega, hbar, gamma)
               for i in range(noscillators)]

    h = np.kron(systems[0].get_ham(), systems[1].get_identity()) \
        + np.kron(systems[0].get_identity(), systems[1].get_ham()) \
        + np.kron(systems[0].get_a(), systems[1].get_adag()) * kappa \
        + np.kron(systems[0].get_adag(), systems[1].get_a()) * kappa

    u = expm(-1j * h)
    print(u)


if __name__ == '__main__':
    main()
