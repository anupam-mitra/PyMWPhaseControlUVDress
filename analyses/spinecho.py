import numpy as np

from numpy import pi

from core import fidelity
from core import spinoperators


def spin_echo_unitary():
    spin = spinoperators.TwoQubitSpin()
    sequence = [
        spin.calc_xysu2(pi/2, 0),
        spin.calc_zrotatetwist(0, pi/2),
        spin.calc_xysu2(pi, 0),
        spin.calc_zrotatetwist(0, pi/2),
        spin.calc_xysu2(pi/2, 0),
    ]

    u = np.eye(4, dtype=complex)
    for step in sequence:
        u = np.dot(step, u)
    return u


def main():
    spin = spinoperators.TwoQubitSpin()
    u = spin_echo_unitary()
    target = spin.calc_yrotatetwist(0, pi)

    print('Microwave spin echo protocol yields')
    print(np.round(u, 3))
    print('Molmer Sorenson y unitary')
    print(np.round(target, 3))
    print('Fidelity between these')
    print(fidelity.unitary_fidelity(target, u))


if __name__ == '__main__':
    main()
