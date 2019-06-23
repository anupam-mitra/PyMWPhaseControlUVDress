import numpy as np


class HarmonicOscillatorTwoBodyCoupling:
    """
    TwoBody Coupling of multiple harmonic oscillators
    """
    def __init__ (self, noscillators, couplingenergies):
        self.noscillators = noscillators
        self.couplingenergies = couplingenergies
