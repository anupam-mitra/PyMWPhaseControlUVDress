def fidelity_twist (theta2):
    fidelity = 1/2 * (1 + cos(theta2/2))
    return fidelity

def fidelity_twistrotate (theta2, theta1):
    fidelity = 1/4 * (1 + cos(theta1)**2 + 2*cos(theta1) * cos(theta2/2))
    return fidelity
