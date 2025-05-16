"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def model(t, vars, params):
    """
    ODE model for the pharmacokinetics of a drug in the brain.
    
    Parameters:
    t : float
        Time variable.
    vars : list
        List of state variables [Cp, Ccsf, Csn].
    params : list
        List of parameters [kel, kmet, kf, kr, kSN].
    
    Returns:
    dvars : list
        List of derivatives [dCp/dt, dCcsf/dt, dCsn/dt].
    """
    Cp, Ccsf, Csn = vars
    kel, kmet, kf, kr, kSN = params

    dCp = -(kel + kmet + kf) * Cp + kr * Ccsf
    dCcsf = kf * Cp - kr * Ccsf - kSN * Csn - kmet * Ccsf
    dCsn = kSN * Ccsf - kmet * Csn

    return [dCp, dCcsf, dCsn]

def simulate_model(init, params):
    """
    Simulate the ODE model using scipy's solve_ivp.

    Parameters:
    init : list
        Initial conditions [Cp0, Ccsf0, Csn0].
    params : list
        Model parameters [kel, kmet, kf, kr, kSN].

    Returns:
    result : OdeResult
        The result of the ODE simulation.
    """

    t = (0, 50)  # Time span for the simulation
    t_eval = np.linspace(t[0], t[1], 10000)  # Time points to evaluate the solution

    sol = solve_ivp(model, t, init, args=(params,), method='RK45', t_eval=t_eval)

    return sol

# Parameter definitions (rate constants)
kel = 0 # Rate of elimination
kmet = 0.1 # Rate of metabolism
kf = 1 # Rate across the BBB (plasma to CSF)
kr = 0.01 # Rate across the BBB (CSF to plasma)
kSN = 0.1 # Rate of transfer from CSF to substantia nigra

# Initial conditions
Cp0 = 100.0 # Initial concentration in plasma (bolus amount)
Ccsf0 = 0.0 # Initial concentration in CSF
Csn0 = 0.0 # Initial concentration in substantia nigra

sol = simulate_model([Cp0, Ccsf0, Csn0], [kel, kmet, kf, kr, kSN])

# Plotting the results
plt.plot(sol.t, sol.y[0], label='Cp (Plasma)')
plt.plot(sol.t, sol.y[1], label='Ccsf (CSF)')
plt.plot(sol.t, sol.y[2], label='Csn (Substantia Nigra)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.show()