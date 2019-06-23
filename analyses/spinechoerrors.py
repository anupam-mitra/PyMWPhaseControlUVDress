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

theta2 = np.pi
theta1 = -np.pi/2

uuv = lambda theta1, theta2: expm(-1j * theta1 * sz - 1j * theta2 * szsquared/2)
umw = lambda theta, phi: expm(-1j * theta * (cos(phi)*sx + sin(phi)*sy))

# Using two random numbers for the two uv dressing and undressing stages
randNumber = np.random.randn(2)*0.02
print(randNumber)

sequence = [\
      umw(pi/2, 0), \
      uuv(theta1 * (1 + randNumber[0]), theta2/2 * (1 + randNumber[0])), \
      umw(pi, 0), \
      uuv(theta1* (1 + randNumber[1]), theta2/2* (1 + randNumber[1])), \
      umw(pi/2, 0), \
]

u = np.eye(4, dtype=complex)
for s in range(len(sequence)):
      u = np.dot(sequence[s], u)

umsy = expm(-1j * pi * sysquared/2)

print('Microwave spin echo protocol yields')
print(np.round(u, 3))
print('Molmer Sorenson y unitary')
print(np.round(umsy, 3))
print('Fidelity between these')
print(fidelity(umsy, u))
