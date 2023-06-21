import numpy as np

class ControlAnsatz:
    """Base class for control ansätze."""
    def evaluate(self, t):
        raise NotImplementedError
    
    def gradient(self, t):
        """Returns gradient w.r.t. parameters."""
        raise NotImplementedError

class GaussianPulse(ControlAnsatz):
    def __init__(self, amplitude, center, width):
        self.amplitude = amplitude
        self.center = center
        self.width = width

    def evaluate(self, t):
        return self.amplitude * np.exp(-(t - self.center)**2 / (2 * self.width**2))

    def gradient(self, t):
        val = self.evaluate(t)
        # d/dA = exp(...)
        # d/dtc = A * exp(...) * (t-tc)/tw^2
        # d/dtw = A * exp(...) * (t-tc)^2 / tw^3
        return np.array([
            val / self.amplitude if self.amplitude != 0 else 0,
            val * (t - self.center) / (self.width**2),
            val * (t - self.center)**2 / (self.width**3)
        ])

class FourierPulse(ControlAnsatz):
    def __init__(self, components):
        """
        components: list of (amplitude, frequency, phase)
        """
        self.components = components # List of lists/tuples [A, w, phi]

    def evaluate(self, t):
        return sum(A * np.sin(w * t + phi) for A, w, phi in self.components)

    def gradient(self, t):
        grads = []
        for A, w, phi in self.components:
            # d/dA = sin(wt+phi)
            # d/dw = A * t * cos(wt+phi)
            # d/dphi = A * cos(wt+phi)
            grads.append(np.sin(w * t + phi))
            grads.append(A * t * np.cos(w * t + phi))
            grads.append(A * np.cos(w * t + phi))
        return np.array(grads)

class ErfPulse(ControlAnsatz):
    def __init__(self, amplitude, t1, t2, slope):
        self.amplitude = amplitude
        self.t1 = t1
        self.t2 = t2
        self.slope = slope

    def evaluate(self, t):
        # approximation of the Erf-based pulse in the paper
        # la.g(t) = (A/4) * (1 + erf(sqrt(pi)*s/A * (t-t1))) * erfc(sqrt(pi)*s/A * (t-t2))
        from scipy.special import erf, erfc
        s = self.slope
        A = self.amplitude
        arg1 = np.sqrt(np.pi) * s / A * (t - self.t1) if A != 0 else 0
        arg2 = np.sqrt(np.pi) * s / A * (t - self.t2) if A != 0 else 0
        return (A / 4.0) * (1.0 + erf(arg1)) * erfc(arg2)

    def gradient(self, t):
        if self.amplitude == 0:
            return np.zeros(4)

        from scipy.special import erf, erfc

        A = self.amplitude
        s = self.slope
        c = np.sqrt(np.pi) * s

        arg1 = c / A * (t - self.t1)
        arg2 = c / A * (t - self.t2)

        e1 = erf(arg1)
        e2 = erfc(arg2)
        exp1 = np.exp(-arg1**2)
        exp2 = np.exp(-arg2**2)

        dA = 0.25 * (1.0 + e1) * e2 + (1.0 / (2.0 * np.sqrt(np.pi))) * (
            -arg1 * exp1 * e2 + (1.0 + e1) * arg2 * exp2
        )
        dt1 = -0.5 * s * exp1 * e2
        dt2 = 0.5 * s * (1.0 + e1) * exp2
        ds = 0.5 * ((t - self.t1) * exp1 * e2 - (t - self.t2) * (1.0 + e1) * exp2)

        return np.array([dA, dt1, dt2, ds])

def sigmoid_window(t, T_final, g=40, delta_tau=0.075):
    """
    Window function to ensure smooth start and finish.
    S_up(t) * S_down(t)
    """
    tau = t / T_final
    # Ascending
    s_down_start = 1.0 / (1.0 + np.exp(4 * g * (tau - delta_tau)))
    s_up_start = 1.0 - s_down_start
    # Descending
    s_down_end = 1.0 / (1.0 + np.exp(4 * g * (tau - (1.0 - delta_tau))))
    
    return s_up_start * s_down_end

def amplitude_bound(val, a, b):
    """
    Bounds value to [a, b] using a scaled sine.
    C(x, a, b) = ((b-a)/2) * sin((x - (b+a)/2) / ((b-a)/2)) + (b+a)/2
    """
    mid = (a + b) / 2.0
    half_width = (b - a) / 2.0
    return half_width * np.sin((val - mid) / half_width) + mid
