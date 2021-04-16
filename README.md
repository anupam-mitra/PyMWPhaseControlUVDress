# quantumcontrol.py

Implementing quantum optimal control in Python using `scipy` and `numpy`.

There are several ingredients to quantum optimal control

1. **State space**. The spaces of states where the dynamics is occuring.
2. **Dynamical operators**. Generating operators which generate the dynamics, for example the Hamiltonian for closed quantum systems
jump operators for the open quantu, systems under the Markovian approximation.
3. **Differential equation solution**. Solving the dynamical equation, for example time dependent Schrödinger equation
for closed quantum system, Lindblad Master equation for open quantum systems under the Markovian approximation.
4. **Cost function evalation**.
    1. How close the the achieved state to the target?
    2. How well do the control fields meet the constraints?
    3. How well the intermediate dynamics meets its requirements, for example adiabaticity?
5. **Control field generation**. Generating the control fields from their specification, for example in the time-domain or using a set
of basis functions.
6. **Optimization of control fields**. Updating the control fields to iterate to the optimal solution for the control problem.
