import numpy as np

from numpy import pi

import costfunctions
import robustcostfunctions
import rydbergatoms


def hamiltonian_parameters(OmegaR=1, OmegaMW=1, DeltaR=0.01, DeltaMW=0.01,
                           deltaRa=0, deltaRb=0):
    return {
        'OmegaRa': OmegaR,
        'OmegaRb': OmegaR,
        'OmegaMWa': OmegaMW,
        'OmegaMWb': OmegaMW,
        'DeltaRa': DeltaR + deltaRa,
        'DeltaRb': DeltaR + deltaRb,
        'DeltaMWa': DeltaMW,
        'DeltaMWb': DeltaMW,
    }


def base_hamiltonian_parameters(OmegaR=1, OmegaMW=1, DeltaR=0.01, DeltaMW=0.01):
    return {
        'OmegaR': OmegaR,
        'OmegaMW': OmegaMW,
        'DeltaR': DeltaR,
        'DeltaMW': DeltaMW,
    }


def phase_propagator_parameters(hamiltonian_params, nsteps_pi_pulse, n_pi_pulses):
    return {
        'HamiltonianParameters': hamiltonian_params,
        'HamiltonianMatrix': rydbergatoms.hamiltonian_PerfectBlockade,
        'HamiltonianMatrixGradient': rydbergatoms.hamiltonian_grad_PerfectBlockade,
        'Nsteps': n_pi_pulses * nsteps_pi_pulse,
        'Tstep': pi / nsteps_pi_pulse,
        'Tcontrol': n_pi_pulses * pi,
    }


def controlled_z_unitary():
    return rydbergatoms.ket_00 * rydbergatoms.bra_00 \
         + rydbergatoms.ket_01 * rydbergatoms.bra_01 \
         + rydbergatoms.ket_10 * rydbergatoms.bra_10 \
         - rydbergatoms.ket_11 * rydbergatoms.bra_11


def state_control_problem(hamiltonian_params, propagator_params,
                          initialization='Constant', robust=False,
                          base_params=None, uncertain_params=None):
    module = robustcostfunctions if robust else costfunctions
    control_problem = {
        'ControlTask': 'StateToStateMap',
        'Initialization': initialization,
        'PropagatorParameters': propagator_params,
        'PureStateInitial': rydbergatoms.ket_00,
        'PureStateTarget': rydbergatoms.target_dressed_state(hamiltonian_params),
        'CostFunction': module.infidelity,
        'CostFunctionGrad': module.infidelity_gradient,
    }

    if robust:
        control_problem['HamiltonianBaseParameters'] = base_params
        control_problem['HamiltonianUncertainParameters'] = uncertain_params

    return control_problem


def unitary_control_problem(propagator_params, target=None, initialization='Random',
                            robust=False, base_params=None, uncertain_params=None,
                            dress_target=False):
    module = robustcostfunctions if robust else costfunctions
    if target is None:
        target = controlled_z_unitary()

    if dress_target:
        h_params = propagator_params['HamiltonianParameters']
        u_dress = rydbergatoms.dressing_unitary(h_params)
        target = np.dot(u_dress, np.dot(target, u_dress.transpose().conjugate()))

    control_problem = {
        'ControlTask': 'UnitaryMap',
        'Initialization': initialization,
        'UnitaryTarget': target,
        'PropagatorParameters': propagator_params,
        'CostFunction': module.infidelity_unitary,
        'CostFunctionGrad': module.infidelity_unitary_gradient,
    }

    if robust:
        control_problem['HamiltonianBaseParameters'] = base_params
        control_problem['HamiltonianUncertainParameters'] = uncertain_params

    return control_problem
