import numpy as np

from numpy import sqrt

dagger = lambda u: np.conjugate(np.transpose(u))

def angularmomentumop(s):
    """
    Generates angular momentum operators
    for spin s
    """

    nstates = 2*s+1

    sz = np.diag([m for m in np.arange(s, -s-1, -1)])

    #Splus = np.diag([sqrt(s*(s+1) - m*(m+1)) for m in np.arange(s, -s, -1)], -1)
    #Sminus = dagger(Splus)

    sminus = np.diag([sqrt(s*(s+1) - m*(m-1)) for m in np.arange(s, -s, -1)], 1)
    splus = dagger(sminus)

    sx = (splus + sminus) * 1/2
    sy = (splus - sminus) * (-1j) * 1/2

    return sx, sy, sz
