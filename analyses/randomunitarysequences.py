import numpy as np
import sys
import scipy as sp

from scipy.linalg import expm
from numpy import sin, cos, exp, sqrt, pi

sys.path.append('../bench/models')

dagger = lambda u: np.transpose(np.conjugate(u))

fidelity = lambda u, v: np.abs(np.trace(np.dot(\
                        dagger(u), v)))**2 / \
                        np.min([np.linalg.matrix_rank(u), np.linalg.matrix_rank(v)])**2

import angularmomentum

sigmax, sigmay, sigmaz = \
    (2*s for s in angularmomentum.angularmomentumop(1/2))

identity2 = np.eye(2)

sx = (np.kron(identity2, sigmax)
      + np.kron(sigmax, identity2)) * 1/2

sy = (np.kron(identity2, sigmay)
      + np.kron(sigmay, identity2)) * 1/2

sz = (np.kron(identity2, sigmaz)
      + np.kron(sigmaz, identity2)) * 1/2

sxsquared = np.dot(sx, sx)
sysquared = np.dot(sy, sy)
szsquared = np.dot(sz, sz)

uuv = lambda theta1, theta2: expm(-1j * theta1 * sz - 1j * theta2 * szsquared/2)
umw = lambda theta, phi: expm(-1j * theta * (cos(phi)*sx + sin(phi)*sy))

# Twist angle for ultraviolet dressing and undressing
thetauv2 = np.pi
# Rotation angle for ultraviolet dressing and undressing
thetauv1 = -np.pi/2

# Rotation angle for microwave pulse
thetamwmax = np.pi
# Rotation direction for microwave pulse
phimwmax = 2*np.pi

nuvstages = 1024
nmwstages = nuvstages + 1

tempscaled = 0.1
randnumuv = np.random.randn(nuvstages, 2) * 0.05
randnummw = np.random.rand(nmwstages, 2)

u = umw(thetamwmax * randnummw[-1, 0], phimwmax * randnummw[-1, 1])
for s in range(nuvstages):
      uuvstep = uuv(thetauv1 * randnumuv[s, 0], thetauv2 * randnumuv[s, 1])
      umwstep = umw(thetamwmax * randnummw[s, 0], phimwmax * randnummw[s, 1])
      u = np.dot(uuvstep, u)
      u = np.dot(umwstep, u)

umsy = expm(-1j * pi * sysquared/2)

print('Microwave spin echo protocol yields')
print(np.round(u, 3))
print('Molmer Sorenson y unitary')
print(np.round(umsy, 3))
print('Fidelity between these')
print(fidelity(umsy, u))


def wootersentropy (u):
      prob = np.abs(u)**2
      logprob = np.log(prob)
      we = -np.sum(prob * logprob)/np.linalg.matrix_rank(u)
      return we

print('WootersEntropy')
print(wootersentropy(u))
