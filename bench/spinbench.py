import numpy as np
import scipy
import angularmomentum

from numpy import sqrt, exp, pi
from scipy.linalg import expm

spin = 1024
nstates = 2*spin+1
niter = 2

randNumbers = np.random.randn(niter)

Sx, Sy, Sz = angularmomentum.angularmomentumop(spin)
identity = np.eye(nstates)

for iter in range(niter):
    print("Starting iteration", iter)
    Uy = expm(-1j * Sy)
    Uz = expm(-1j * Sz/nstates * randNumbers[iter])
    Kmeasure = expm(- (Sz - randNumbers[iter]*identity)**2/(2*spin)) * 1/sqrt(4*pi*nstates)
    Kmanuel = np.dot(Uy, np.dot(Uz, Kmeasure))
    print("Finished iteration ", iter)
