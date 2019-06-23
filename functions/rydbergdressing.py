def perturbed_els_1 (delta_uv_a, delta_uv_b, Omega_uv, Delta_uv):
    """
    Calculates the perturbation to the one atom light shift
    to the dressed states upto second order perturbation theory
    """
    denominator = sqrt(Omega_uv**2 + Delta_uv**2)

    if Delta_uv == 0:
        theta_s = pi/2
    else:
        theta_s = superposing_angle(Omega_uv, Delta_uv)

    els_a_1st = delta_uv_a * sin(theta_s/2)**2
    els_b_1st = delta_uv_b * sin(theta_s/2)**2

    els_a_2nd = 1/denominator * abs(delta_uv_a * cos(theta_s/2) * sin(theta_s/2))**2
    els_b_2nd = 1/denominator * abs(delta_uv_b * cos(theta_s/2) * sin(theta_s/2))**2

    els_a = els_a_1st + els_a_2nd
    els_b = els_b_1st + els_b_2nd

    els_1_err = (els_a + els_b)/2

    return els_1_err

def perturbed_els_2 (delta_uv_a, delta_uv_b, Omega_uv, Delta_uv):
    """
    Calculates the perturbation to the two atom light shift
    to the dressed states upto second order perturbation theory
    """
    denominator = sqrt(2*Omega_uv**2 + Delta_uv**2)

    if Delta_uv == 0:
        theta_s = pi/2
    else:
        theta_s = superposing_angle(sqrt(2)*Omega_uv, Delta_uv)

    delta_uv_cm = delta_uv_a + delta_uv_b
    delta_uv_rel = delta_uv_a - delta_uv_b

    els_2_1st = delta_uv_cm * sin(theta_s/2)**2
    els_2_2nd = 1/denominator * abs(delta_uv_cm**2 * sin(theta_s/2) * cos(theta_s/2))**2 \
            + 2/denominator * abs(delta_uv_rel**2 * sin(theta_s/2))**2

    els_2_err = els_2_1st + els_2_2nd

    return els_2_err

def perturbed_kappa (delta_uv_a, delta_uv_b, Omega_uv, Delta_uv):
    els_1_err = perturbed_els_1(delta_uv_a, delta_uv_b, Omega_uv, Delta_uv)
    els_2_err = perturbed_els_2(delta_uv_a, delta_uv_b, Omega_uv, Delta_uv)

    kappa = els_2_err - 2*els_1_err
    return kappa
