import numpy as np

from numpy import cos, sin

import angularmomentum

class ControllableSpin:
    """
    Represents a spin controllable by a rotation about an axis in the xy plane

    Parameters
    ----------
    s: spin
    """

    def __init__ (self, s):
        self.s = s
        sx, sy, sz = angularmomentum.angularmomentumop(s)


    def hamiltonian_spincontrol (phi):
        """
        Returns the spin control Hamiltonian
        """

        sphi = cos(phi) * self.sx + sin(phi) * self.sy

        return sphi

    def hamiltonian_spincontrol_grad (phi):
        """
        Returns the gradient of the spin control Hamiltonian
        """

        sphi_grad = self.hamiltonian_spincontrol(phi + pi/2)

        return sphi_grad

    def unitary_spincontrol (theta, phi):
        """
        Returns the unitary for a spin control Hamiltonian

        Parameters
        ----------
        theta: angle by which to rotate

        phi: angle of the rotation axis from the x axis
        """

        sphi = self.hamiltonian_spincontrol(phi)
        u = exp(-1j * theta * sphi)
        return u

    def unitary_spincontrol_grad (theta, phi):
        """
        
        Returns the unitary for a spin control Hamiltonian

        Parameters
        ----------
        theta: angle by which to rotate

        phi: angle of the rotation axis from the x axis
        """

        u = self.unitary_spincontrol(theta, phi)
        u_grad = np.dot(np.dot(self.sz, u), self.sz)

        return u_grad
