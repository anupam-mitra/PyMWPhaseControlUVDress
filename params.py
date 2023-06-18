from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List

@dataclass
class PropagatorParameters:
    Nsteps: int
    Tstep: float
    Tcontrol: float
    hamiltonian_params: Any  # Can be a dict or a specific HamiltonianParameters dataclass
    hamiltonian_matrix_func: Callable = None
    hamiltonian_matrix_grad_func: Callable = None

@dataclass
class ControlProblem:
    propagator_params: PropagatorParameters
    cost_function: Optional[Callable] = None
    cost_function_grad: Optional[Callable] = None
    initialization: str = 'Random'
    
    # State targets
    pure_state_initial: Any = None
    pure_state_target: Any = None
    
    # Unitary targets
    unitary_target: Any = None
    n_states_unitary: Optional[int] = None
    
    # Robust control / Inhomogeneities
    hamiltonian_base_params: Any = None
    hamiltonian_uncertain_params: Any = None
    hamiltonian_landmarks: Optional[List[Any]] = None
    landmark_weights: Optional[Any] = None
    
    # Adiabatic / Protocol specific
    unitary_dressing: Any = None
    unitary_undressing: Any = None
    unitary_dressing_landmarks: Optional[List[Any]] = None
    unitary_undressing_landmarks: Optional[List[Any]] = None
    
    # Results
    results: Any = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Helper to convert legacy dictionary to ControlProblem dataclass."""
        prop_dict = d.get('PropagatorParameters', {})
        prop_params = PropagatorParameters(
            Nsteps=prop_dict.get('Nsteps'),
            Tstep=prop_dict.get('Tstep'),
            Tcontrol=prop_dict.get('Tcontrol'),
            hamiltonian_params=prop_dict.get('HamiltonianParameters'),
            hamiltonian_matrix_func=prop_dict.get('HamiltonianMatrix'),
            hamiltonian_matrix_grad_func=prop_dict.get('HamiltonianMatrixGradient')
        )
        
        return cls(
            propagator_params=prop_params,
            cost_function=d.get('CostFunction'),
            cost_function_grad=d.get('CostFunctionGrad'),
            initialization=d.get('Initialization', 'Random'),
            pure_state_initial=d.get('PureStateInitial'),
            pure_state_target=d.get('PureStateTarget'),
            unitary_target=d.get('UnitaryTarget'),
            n_states_unitary=d.get('NStatesUnitary'),
            hamiltonian_base_params=d.get('HamiltonianBaseParameters'),
            hamiltonian_uncertain_params=d.get('HamiltonianUncertainParameters'),
            hamiltonian_landmarks=d.get('HamiltonianLandmarks'),
            landmark_weights=d.get('LandmarkWeights'),
            unitary_dressing=d.get('UnitaryDressing'),
            unitary_undressing=d.get('UnitaryUndressing'),
            unitary_dressing_landmarks=d.get('UnitaryDressingLandmarks'),
            unitary_undressing_landmarks=d.get('UnitaryUnDressingLandmarks'),
            results=d.get('Results')
        )
