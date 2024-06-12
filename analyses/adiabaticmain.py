#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from numpy import sqrt, sign, pi, arctan2, sin

from analyses import adiabaticdressundress


def evaluate_example(numpoints=512):
    adiabatic = adiabaticdressundress.RydbergAdiabaticDressUndress(
        Omega_min=0,
        Omega_max=1,
        Delta_min=0.1,
        Delta_max=2.5,
        t_gaussian_width=12,
        t_constant_duration=4.56267,
        t_mid=0,
        gausswidthslinear=2,
    )

    t = np.linspace(adiabatic.t_dress_begin, adiabatic.t_undress_end, num=numpoints)

    Omega = np.empty(t.shape)
    dOmega_dt = np.empty(t.shape)
    Delta = np.empty(t.shape)
    dDelta_dt = np.empty(t.shape)
    kappa = np.empty(t.shape)
    theta_admix_1 = np.empty(t.shape)
    theta_admix_2 = np.empty(t.shape)

    for n in range(t.shape[0]):
        Omega[n], dOmega_dt[n] = adiabatic.get_Omega(t=t[n])
        Delta[n], dDelta_dt[n] = adiabatic.get_Delta(t[n])
        kappa[n] = -Delta[n]/2 - sign(Delta[n])/2 * (
            sqrt(Delta[n]**2 + 2*Omega[n]**2) - 2*sqrt(Delta[n]**2 + Omega[n]**2))
        theta_admix_1[n] = arctan2(Omega[n], Delta[n])
        theta_admix_2[n] = arctan2(sqrt(2) * Omega[n], Delta[n])

    diabaticity = np.abs((Omega * dDelta_dt - Delta * dOmega_dt)/(Omega**2 + Delta**2)**(3/2))
    integrated_rydberg_population = (
        2/3*scipy.integrate.cumtrapz(sin(theta_admix_1/2)**2, t)
        + 1/3*scipy.integrate.cumtrapz(sin(theta_admix_2/2)**2, t))

    return {
        'adiabatic': adiabatic,
        't': t,
        'Omega': Omega,
        'Delta': Delta,
        'kappa': kappa,
        'theta_admix_1': theta_admix_1,
        'theta_admix_2': theta_admix_2,
        'diabaticity': diabaticity,
        'integrated_rydberg_population': integrated_rydberg_population,
    }


def main():
    data = evaluate_example()
    print('Max diabaticity = %g' % (np.max(data['diabaticity'])))
    print('Integrated Rydberg population = %g' % (data['integrated_rydberg_population'][-1],))

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 32
    plt.rc('text', usetex=True)
    plt.rcParams["figure.figsize"] = [12, 12]

    t = data['t']
    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].plot(t/2/pi, data['Omega'], color='blue')
    axes[1].plot(t/2/pi, data['Delta'], color='blue')
    axes[2].plot(t/2/pi, data['kappa'], color='blue')
    axes[3].plot(t/2/pi, sin(data['theta_admix_1']/2)**2, color='blue')
    axes[3].plot(t/2/pi, sin(data['theta_admix_2']/2)**2, color='red')

    axes[0].set_ylabel(r'$\Omega_{\mathrm{uv}} / \Omega_{\max}$')
    axes[1].set_ylabel(r'$\Delta_{\mathrm{uv}} / \Omega_{\max}$')
    axes[2].set_ylabel(r'$\kappa / \Omega_{\max}$')
    axes[3].set_ylabel(r'$\mathcal{P}_r$')
    axes[-1].set_xlabel(r'$\Omega_{\max} t / (2\pi)$')

    title_string = r'$\frac{1}{\pi}\int\kappa(t)dt = %g$' % (
        scipy.integrate.cumtrapz(data['kappa'], t)[-1]/pi,)
    axes[0].set_title(title_string)


if __name__ == '__main__':
    main()
