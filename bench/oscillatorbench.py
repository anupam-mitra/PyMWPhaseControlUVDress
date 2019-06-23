import numpy as np
import scipy

import harmonicoscillator

from numpy import sqrt
from scipy.linalg import expm

dagger = lambda u: np.conjugate(np.transpose(u))

noscillators = 2

hbar = 1

#angular frequency of each oscillator
omega = 1

#decay rate of each oscillator
gamma = 1/8

#coupling strength between oscillators
kappa = 1/4

nmax = 3

systems = [harmonicoscillator.DampedHarmonicOscillator(nmax, omega, hbar, gamma) \
            for i in range(noscillators)]

h = np.kron(systems[0].get_ham(), systems[1].get_identity()) \
    + np.kron(systems[0].get_identity(), systems[1].get_ham()) \
    + np.kron(systems[0].get_a(), systems[1].get_adag()) * kappa \
    + np.kron(systems[0].get_adag(), systems[1].get_a()) * kappa


u = expm(-1j * h)

print(u)
