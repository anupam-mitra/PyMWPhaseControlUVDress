import numpy as np

from core.physics import hamiltonian_ms_gate, hamiltonian_ms_gate_grad


CURRENT_TCONTROL = 20.0


def one_photon_ramp_val(t, params):
    o_max, d_max, d_min, tw, tc = params
    t2 = -tc / 2.0
    t3 = tc / 2.0
    t1 = -CURRENT_TCONTROL / 2.0
    t4 = CURRENT_TCONTROL / 2.0

    if t < t2:
        delta = d_max + (d_min - d_max) / (t2 - t1) * (t - t1)
    elif t <= t3:
        delta = d_min
    else:
        delta = d_min + (d_max - d_min) / (t4 - t3) * (t - t3)

    if t < t2:
        omega = o_max * np.exp(-(t - t2) ** 2 / (2 * tw ** 2))
    elif t <= t3:
        omega = o_max
    else:
        omega = o_max * np.exp(-(t - t3) ** 2 / (2 * tw ** 2))

    return np.array([omega, delta])


def one_photon_ramp_grad(t, params):
    o_max, d_max, d_min, tw, tc = params
    t2 = -tc / 2.0
    t3 = tc / 2.0
    t1 = -CURRENT_TCONTROL / 2.0
    t4 = CURRENT_TCONTROL / 2.0
    grad = np.zeros((2, 5))

    if t < t2:
        val = np.exp(-(t - t2) ** 2 / (2 * tw ** 2))
        grad[0, 0] = val
        grad[0, 3] = o_max * val * (t - t2) ** 2 / (tw ** 3)
        grad[0, 4] = -o_max * val * (t - t2) / (2.0 * tw ** 2)
    elif t <= t3:
        grad[0, 0] = 1.0
    else:
        val = np.exp(-(t - t3) ** 2 / (2 * tw ** 2))
        grad[0, 0] = val
        grad[0, 3] = o_max * val * (t - t3) ** 2 / (tw ** 3)
        grad[0, 4] = o_max * val * (t - t3) / (2.0 * tw ** 2)

    if t < t2:
        denom = t2 - t1
        frac = (t - t1) / denom
        grad[1, 1] = 1.0 - frac
        grad[1, 2] = frac
        grad[1, 4] = 2.0 * (d_min - d_max) * (t - t1) / (CURRENT_TCONTROL - tc) ** 2
    elif t <= t3:
        grad[1, 2] = 1.0
    else:
        denom = t4 - t3
        frac = (t - t3) / denom
        grad[1, 1] = frac
        grad[1, 2] = 1.0 - frac
        grad[1, 4] = 2.0 * (d_max - d_min) * (t - t4) / (CURRENT_TCONTROL - tc) ** 2

    return grad


def two_photon_ramp_val(t, params):
    o_max, d_max, d_min, tw, tc = params
    t_total = CURRENT_TCONTROL
    t2 = -tc / 2.0
    t3 = tc / 2.0
    t1 = -t_total / 2.0
    t4 = t_total / 2.0

    if t < t2:
        omega = o_max * np.exp(-(t - t2) ** 2 / (2 * tw ** 2))
    elif t <= t3:
        omega = o_max
    else:
        omega = o_max * np.exp(-(t - t3) ** 2 / (2 * tw ** 2))

    if t < t2:
        delta = d_max + (d_min - d_max) * (t - t1) / (t2 - t1) if (t2 - t1) != 0 else d_max
    elif t <= t3:
        delta = d_min
    else:
        delta = d_min + (d_max - d_min) * (t - t3) / (t4 - t3) if (t4 - t3) != 0 else d_min

    return np.array([omega, delta])


def two_photon_ramp_grad(t, params):
    o_max, d_max, d_min, tw, tc = params
    t_total = CURRENT_TCONTROL
    t2 = -tc / 2.0
    t3 = tc / 2.0
    t1 = -t_total / 2.0
    t4 = t_total / 2.0
    grad = np.zeros((2, 5))

    if t < t2:
        val = np.exp(-(t - t2) ** 2 / (2 * tw ** 2))
        grad[0, 0] = val
        grad[0, 3] = o_max * val * (t - t2) ** 2 / (tw ** 3)
        grad[0, 4] = -o_max * val * (t - t2) / (2.0 * tw ** 2)
    elif t <= t3:
        grad[0, 0] = 1.0
    else:
        val = np.exp(-(t - t3) ** 2 / (2 * tw ** 2))
        grad[0, 0] = val
        grad[0, 3] = o_max * val * (t - t3) ** 2 / (tw ** 3)
        grad[0, 4] = o_max * val * (t - t3) / (2.0 * tw ** 2)

    if t < t2:
        denom = (t2 - t1)
        if denom != 0:
            frac = (t - t1) / denom
            grad[1, 1] = 1.0 - frac
            grad[1, 2] = frac
            grad[1, 4] = 2.0 * (d_min - d_max) * (t - t1) / (t_total - tc) ** 2
    elif t <= t3:
        grad[1, 2] = 1.0
    else:
        denom = (t4 - t3)
        if denom != 0:
            frac = (t - t3) / denom
            grad[1, 1] = frac
            grad[1, 2] = 1.0 - frac
            grad[1, 4] = 2.0 * (d_max - d_min) * (t - t4) / (t_total - tc) ** 2

    return grad


def make_ms_hamiltonian(phi_val, h_params):
    return hamiltonian_ms_gate(phi_val, h_params)


def make_ms_hamiltonian_grad(t, phi_val, h_params, params_vec):
    dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, h_params)
    dphi_dalpha = one_photon_ramp_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    return [dphi_dalpha[0, k] * dH_domega + dphi_dalpha[1, k] * dH_ddelta for k in range(len(params_vec))]


def make_ms_hamiltonian_grad_2p(t, phi_val, h_params, params_vec):
    dH_domega, dH_ddelta = hamiltonian_ms_gate_grad(phi_val, h_params)
    dphi_dalpha = two_photon_ramp_grad(t - CURRENT_TCONTROL / 2.0, params_vec)
    return [dphi_dalpha[0, k] * dH_domega + dphi_dalpha[1, k] * dH_ddelta for k in range(len(params_vec))]


class GaussianCosineAnsatz:
    def __init__(self, T_control, dim, params, tc=None, w=None):
        self.T_control = T_control
        self.dim = dim
        self.params = params
        self.tc = tc if tc is not None else T_control / 2.0
        self.w = w if w is not None else T_control / 4.0

    def evaluate(self, t):
        gauss = np.exp(-(t - self.tc) ** 2 / (2 * self.w ** 2))
        val = 0.0
        for n in range(1, self.dim + 1):
            omega_n = 2 * np.pi * np.sqrt(n) / self.T_control
            A_n = self.params[2 * (n - 1)]
            B_n = self.params[2 * (n - 1) + 1]
            val += A_n * np.cos(omega_n * t) + B_n * np.sin(omega_n * t)
        return gauss * val

    def gradient(self, t):
        gauss = np.exp(-(t - self.tc) ** 2 / (2 * self.w ** 2))
        grads = np.zeros(2 * self.dim)
        for n in range(1, self.dim + 1):
            omega_n = 2 * np.pi * np.sqrt(n) / self.T_control
            grads[2 * (n - 1)] = gauss * np.cos(omega_n * t)
            grads[2 * (n - 1) + 1] = gauss * np.sin(omega_n * t)
        return grads
