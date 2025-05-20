"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.figure as pltf

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
    kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn = params

    dCp = -(kcl + kmetp + kfcsf) * Cp + krcsf * Ccsf
    dCcsf = kfcsf * Cp - (krcsf + kmetcsf + kfsn) * Ccsf + krsn * Csn
    dCsn = kfsn * Ccsf - krsn * Csn

    return [dCp, dCcsf, dCsn]

def simulate_model(init, params, time_end):
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

    t = (0, time_end)  # Time span for the simulation
    t_eval = np.linspace(t[0], t[1], 10000)  # Time points to evaluate the solution

    sol = solve_ivp(model, t, init, args=(params,), method='RK45', t_eval=t_eval)

    return sol

kcl = 0.0001 # Rate of elimination (1/min)
kmetp = 0 # Rate of metabolism
kfcsf = 0.002 # Rate across the BBB (plasma to CSF)
krcsf = 0.0002 # Rate across the BBB (CSF to plasma)
kmetcsf = 0.03 # Rate of metabolism in CSF (1/min)
kfsn = 0.1 # Rate of transfer from CSF to substantia nigra
krsn = 0.05 # Rate of transfer from substantia nigra to CSF

Cp0 = 100 # Initial concentration in plasma ug/mL
Ccsf0 = 0.0 # Initial concentration in CSF
Csn0 = 0.0 # Initial concentration in substantia nigra

sol = simulate_model([Cp0, Ccsf0, Csn0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=10)

plt.plot(sol.t, sol.y[0], label='Cp (Plasma)')
plt.plot(sol.t, sol.y[1], label='Ccsf (CSF)')
plt.plot(sol.t, sol.y[2], label='Csn (Substantia Nigra)')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (ug/L)')
plt.legend()
plt.show()

# stim_times = [0.00001, 0.01, 0.05, 0.1, 0.5, 1, 5]

# for i, stim_time_i in enumerate(stim_times):
#     # Stim time
#     stim_time = stim_time_i

#     # Parameter definitions (rate constants)
#     kcl = 0.25 # Rate of elimination (1/min)
#     kmetp = 0 # Rate of metabolism
#     kfcsf = 0.02 # Rate across the BBB (plasma to CSF)
#     krcsf = 0.002 # Rate across the BBB (CSF to plasma)
#     kmetcsf = 0.1 # Rate of metabolism in CSF (1/min)
#     kfsn = 0.1 # Rate of transfer from CSF to substantia nigra
#     krsn = 0.05 # Rate of transfer from substantia nigra to CSF

#     # Initial conditions
#     Cp0 = 100 # Initial concentration in plasma ug/mL
#     Ccsf0 = 0.0 # Initial concentration in CSF
#     Csn0 = 0.0 # Initial concentration in substantia nigra

#     sol_0 = simulate_model([Cp0, Ccsf0, Csn0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=0.1)

#     Cp0_0 = sol_0.y[0, -1]
#     Ccsf0_0 = sol_0.y[1, -1]
#     Csn0_0 = sol_0.y[2, -1]
#     kfcsf = 2 # Rate across the BBB (plasma to CSF)
#     krcsf = 0.2 # Rate across the BBB (CSF to plasma)

#     sol_1 = simulate_model([Cp0_0, Ccsf0_0, Csn0_0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=stim_time)

#     Cp0_1 = sol_1.y[0, -1]
#     Ccsf0_1 = sol_1.y[1, -1]
#     Csn0_1 = sol_1.y[2, -1]
#     kfcsf = 0.02
#     krcsf = 0.002

#     sol_2 = simulate_model([Cp0_1, Ccsf0_1, Csn0_1], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=150)

#     t_0 = sol_0.t
#     t_1 = sol_1.t + 0.1
#     t_2 = sol_2.t + stim_time + 0.1
#     t = np.concatenate([t_0, t_1, t_2])
#     y = np.concatenate([sol_0.y, sol_1.y, sol_2.y], axis=1)

#     plt.subplot(len(stim_times), 2, 2*(i)+1)
#     plt.plot(t, y[0], label='Cp (Plasma)')
#     plt.plot(t, y[1], label='Ccsf (CSF)')
#     plt.plot(t, y[2], label='Csn (Substantia Nigra)')
#     plt.xlabel('Time (min)')
#     plt.ylabel('Concentration (ug/L)')
#     # plt.ylim(0,2)
#     plt.xlim(0,5)
#     plt.grid()

#     plt.subplot(len(stim_times), 2, 2*(i+1))
#     plt.plot(t, y[0], label='Cp (Plasma)')
#     plt.plot(t, y[1], label='Ccsf (CSF)')
#     plt.plot(t, y[2], label='Csn (Substantia Nigra)')
#     plt.xlabel('Time (min)')
#     plt.ylabel('Concentration (ug/L)')
#     plt.ylim(0,15)
#     # plt.xlim(0,5)
#     plt.grid()

# plt.tight_layout()
# plt.show()


# # Parameter definitions (rate constants)
# kcl = 0.0001 # Rate of elimination (1/min)
# kmetp = 0 # Rate of metabolism
# kfcsf = 0.002 # Rate across the BBB (plasma to CSF)
# krcsf = 0.0002 # Rate across the BBB (CSF to plasma)
# kmetcsf = 0.03 # Rate of metabolism in CSF (1/min)
# kfsn = 0.1 # Rate of transfer from CSF to substantia nigra
# krsn = 0.05 # Rate of transfer from substantia nigra to CSF

# stim_time = 0.25

# Cp0 = 100 # Initial concentration in plasma ug/mL
# Ccsf0 = 0.0 # Initial concentration in CSF
# Csn0 = 0.0 # Initial concentration in substantia nigra

# sol_0 = simulate_model([Cp0, Ccsf0, Csn0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=10)

# Cp0_0 = sol_0.y[0, -1]
# Ccsf0_0 = sol_0.y[1, -1]
# Csn0_0 = sol_0.y[2, -1]
# kfcsf = 1 # Rate across the BBB (plasma to CSF)
# krcsf = 0.1 # Rate across the BBB (CSF to plasma)

# sol_1 = simulate_model([Cp0_0, Ccsf0_0, Csn0_0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=stim_time)

# Cp0_1 = sol_1.y[0, -1]
# Ccsf0_1 = sol_1.y[1, -1]
# Csn0_1 = sol_1.y[2, -1]
# kfcsf = 0.002
# krcsf = 0.0002

# sol_2 = simulate_model([Cp0_1, Ccsf0_1, Csn0_1], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=250)

# Cp0_2 = sol_2.y[0, -1]
# Ccsf0_2 = sol_2.y[1, -1]
# Csn0_2 = sol_2.y[2, -1]
# kfcsf = 1
# krcsf = 0.1

# sol_3 = simulate_model([Cp0_2, Ccsf0_2, Csn0_2], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=stim_time)

# Cp0_3 = sol_3.y[0, -1]
# Ccsf0_3 = sol_3.y[1, -1]
# Csn0_3 = sol_3.y[2, -1]
# kfcsf = 0.002
# krcsf = 0.0002
# sol_4 = simulate_model([Cp0_3, Ccsf0_3, Csn0_3], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=250)
# Cp0_4 = sol_4.y[0, -1]
# Ccsf0_4 = sol_4.y[1, -1]
# Csn0_4 = sol_4.y[2, -1]
# kfcsf = 1
# krcsf = 0.1
# sol_5 = simulate_model([Cp0_4, Ccsf0_4, Csn0_4], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=stim_time)
# Cp0_5 = sol_5.y[0, -1]
# Ccsf0_5 = sol_5.y[1, -1]
# Csn0_5 = sol_5.y[2, -1]
# kfcsf = 0.002
# krcsf = 0.0002
# sol_6 = simulate_model([Cp0_5, Ccsf0_5, Csn0_5], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=250)

# t_shift = 0
# t_0 = sol_0.t + t_shift
# t_shift += 10
# t_1 = sol_1.t + t_shift
# t_shift += stim_time
# t_2 = sol_2.t + t_shift
# t_shift += 250
# t_3 = sol_3.t + t_shift
# t_shift += stim_time
# t_4 = sol_4.t + t_shift
# t_shift += 250
# t_5 = sol_5.t + t_shift
# t_shift += stim_time
# t_6 = sol_6.t + t_shift
# t = np.concatenate([t_0, t_1, t_2, t_3, t_4, t_5, t_6])
# y = np.concatenate([sol_0.y, sol_1.y, sol_2.y, sol_3.y, sol_4.y, sol_5.y, sol_6.y], axis=1)

# plt.plot(t, y[0], label='Cp (Plasma)')
# plt.plot(t, y[1], label='Ccsf (CSF)')
# plt.plot(t, y[2], label='Csn (Substantia Nigra)')
# plt.vlines([10, 260.25, 510.5], ymin=0, ymax=100, colors='r', linestyles='dashed')
# plt.xlabel('Time (min)')
# plt.ylabel('Concentration (ug/L)')
# plt.legend()
# plt.grid()
# plt.show()



stim_strengths = [1, 10, 100, 1000]

for i, stim_strength_i in enumerate(stim_strengths):
    # Stim strength
    stim_strength = stim_strength_i

    # Parameter definitions (rate constants)
    kcl = 0.25 # Rate of elimination (1/min)
    kmetp = 0 # Rate of metabolism
    kfcsf = 0.02 # Rate across the BBB (plasma to CSF)
    krcsf = 0.002 # Rate across the BBB (CSF to plasma)
    kmetcsf = 0.1 # Rate of metabolism in CSF (1/min)
    kfsn = 0.1 # Rate of transfer from CSF to substantia nigra
    krsn = 0.05 # Rate of transfer from substantia nigra to CSF

    # Initial conditions
    Cp0 = 100 # Initial concentration in plasma ug/mL
    Ccsf0 = 0.0 # Initial concentration in CSF
    Csn0 = 0.0 # Initial concentration in substantia nigra

    sol_0 = simulate_model([Cp0, Ccsf0, Csn0], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=0.1)

    Cp0_0 = sol_0.y[0, -1]
    Ccsf0_0 = sol_0.y[1, -1]
    Csn0_0 = sol_0.y[2, -1]
    kfcsf_stim = kfcsf * stim_strength # Rate across the BBB (plasma to CSF)
    krcsf_stim = krcsf * stim_strength # Rate across the BBB (CSF to plasma)

    sol_1 = simulate_model([Cp0_0, Ccsf0_0, Csn0_0], [kcl, kmetp, kfcsf_stim, krcsf_stim, kmetcsf, kfsn, krsn], time_end=0.25)

    Cp0_1 = sol_1.y[0, -1]
    Ccsf0_1 = sol_1.y[1, -1]
    Csn0_1 = sol_1.y[2, -1]
    kfcsf = 0.02
    krcsf = 0.002

    sol_2 = simulate_model([Cp0_1, Ccsf0_1, Csn0_1], [kcl, kmetp, kfcsf, krcsf, kmetcsf, kfsn, krsn], time_end=150)

    t_0 = sol_0.t
    t_1 = sol_1.t + 0.1
    t_2 = sol_2.t + 0.25 + 0.1
    t = np.concatenate([t_0, t_1, t_2])
    y = np.concatenate([sol_0.y, sol_1.y, sol_2.y], axis=1)

    plt.subplot(len(stim_strengths), 2, 2*(i)+1)
    plt.plot(t, y[0], label='Cp (Plasma)')
    plt.plot(t, y[1], label='Ccsf (CSF)')
    plt.plot(t, y[2], label='Csn (Substantia Nigra)')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (ug/L)')
    # plt.ylim(0,2)
    plt.xlim(0,5)
    plt.grid()

    plt.subplot(len(stim_strengths), 2, 2*(i+1))
    plt.plot(t, y[0], label='Cp (Plasma)')
    plt.plot(t, y[1], label='Ccsf (CSF)')
    plt.plot(t, y[2], label='Csn (Substantia Nigra)')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (ug/L)')
    # plt.ylim(0,15)
    # plt.xlim(0,5)
    plt.grid()

plt.tight_layout()
plt.show()