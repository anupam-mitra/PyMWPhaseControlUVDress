import numpy as np

from numpy import pi

import fidelity
import spinoperators


def random_sequence_unitary(nuvstages=1024, uv_error_width=0.05):
    spin = spinoperators.TwoQubitSpin()
    randnumuv = np.random.randn(nuvstages, 2) * uv_error_width
    randnummw = np.random.rand(nuvstages + 1, 2)

    u = spin.calc_xysu2(pi * randnummw[-1, 0], 2*pi * randnummw[-1, 1])
    for s in range(nuvstages):
        u = np.dot(spin.calc_zrotatetwist(-pi/2 * randnumuv[s, 0], pi * randnumuv[s, 1]), u)
        u = np.dot(spin.calc_xysu2(pi * randnummw[s, 0], 2*pi * randnummw[s, 1]), u)
    return u


def wootersentropy(u):
    prob = np.abs(u)**2
    logprob = np.log(prob)
    return -np.sum(prob * logprob)/np.linalg.matrix_rank(u)


def main():
    spin = spinoperators.TwoQubitSpin()
    u = random_sequence_unitary()
    target = spin.calc_yrotatetwist(0, pi)

    print('Random sequence yields')
    print(np.round(u, 3))
    print('Molmer Sorenson y unitary')
    print(np.round(target, 3))
    print('Fidelity between these')
    print(fidelity.unitary_fidelity(target, u))
    print('WootersEntropy')
    print(wootersentropy(u))


if __name__ == '__main__':
    main()
