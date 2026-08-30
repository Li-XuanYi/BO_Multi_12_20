"""Debug script to find the exact error location"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pybamm
    print(f"PyBaMM version: {pybamm.__version__}")
except ImportError:
    print("PyBaMM not installed")
    sys.exit(1)

# Test basic simulation
print("\nTesting basic PyBaMM SPMe simulation...")

options = {"thermal": "lumped"}
model = pybamm.lithium_ion.SPMe(options=options)
param = pybamm.ParameterValues("Chen2020")

# Update with SPM.py identified parameters
param.update({
    "Current function [A]": "[input]",
    "Upper voltage cut-off [V]": 4.4,
    'Negative particle radius [m]': 4.69e-06,
    'Negative electrode active material volume fraction': 0.73,
    'Negative electrode conductivity [S.m-1]': 258.00,
    'Negative electrode diffusivity [m2.s-1]': 3.96e-14,
    'Positive particle radius [m]': 4.17e-06,
    'Positive electrode active material volume fraction': 0.66,
    'Positive electrode conductivity [S.m-1]': 0.22,
    'Positive electrode diffusivity [m2.s-1]': 4.80e-15,
    'Total heat transfer coefficient [W.m-2.K-1]': 17.36,
    'Separator specific heat capacity [J.kg-1.K-1]': 2905.50,
    'Negative electrode specific heat capacity [J.kg-1.K-1]': 2400.56,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 2715.82,
    'Negative current collector specific heat capacity [J.kg-1.K-1]': 1138.79,
    'Positive current collector specific heat capacity [J.kg-1.K-1]': 1252.81,
})

SOH = 1.0
param.update({
    'Maximum concentration in negative electrode [mol.m-3]': SOH * (33133 - 1308) + 1308
})
param.update({
    'Nominal cell capacity [A.h]': 5.0 * SOH
})

param["Initial temperature [K]"] = 298.15
param["Ambient temperature [K]"] = 298.15

try:
    param.set_initial_stoichiometries("2.8 V")
    print("✓ Initial stoichiometries set successfully")
except Exception as e:
    print(f"✗ Failed to set initial stoichiometries: {e}")

# Try a simple simulation
print("\nRunning a simple simulation...")
sim = pybamm.Simulation(model, parameter_values=param)
t_eval = np.linspace(0, 60, 61)  # 60 seconds

try:
    sol = sim.solve(t_eval, inputs={"Current function [A]": -5.0})
    print("✓ Simulation completed successfully")

    v = sol["Voltage [V]"].entries
    t = sol["X-averaged cell temperature [K]"].entries
    c = sol["R-averaged negative particle concentration [mol.m-3]"].entries

    print(f"  Voltage entries shape: {np.asarray(v).shape}")
    print(f"  Temperature entries shape: {np.asarray(t).shape}")
    print(f"  Concentration entries shape: {np.asarray(c).shape}")

    # Check if we can convert to float
    print("\nTesting array conversion...")
    v_arr = np.asarray(v, dtype=float).reshape(-1)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    print(f"  Reshaped voltage: {v_arr.shape}, values range [{v_arr.min():.3f}, {v_arr.max():.3f}]")
    print(f"  Reshaped temperature: {t_arr.shape}, values range [{t_arr.min():.2f}, {t_arr.max():.2f}]")

except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
