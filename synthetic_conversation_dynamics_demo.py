"""
# Copyright (c) 2025 Kaleb J. Wasierski
# Licensed for non‑commercial use only — see LICENSE.txt in the repo root.

synthetic_conversation_dynamics_demo.py
---------------------------------------
Demo: Comparison of high-intensity vs full-variance conversation dynamics
Simulates Novelty (N), Constraint (C), and Cognitive Load (L) for 20 turns.
All data is synthetic; no personal or sensitive information is used.
"""

import numpy as np
import matplotlib.pyplot as plt

# Number of conversation turns to simulate
turns = 20

# -------------------------------
# Current High-Intensity Pattern (synthetic)
# N, C, L mostly between 3-5 to simulate plateau effect
np.random.seed(42)  # reproducibility
current_N = np.random.randint(3, 6, size=turns)
current_C = np.random.randint(3, 6, size=turns)
current_L = np.random.randint(3, 6, size=turns)

# -------------------------------
# Hypothetical Full-Variance Pattern (synthetic)
# N, C, L vary 0-5 to simulate erratic conversation dynamics
full_var_N = np.random.randint(0, 6, size=turns)
full_var_C = np.random.randint(0, 6, size=turns)
full_var_L = np.random.randint(0, 6, size=turns)

# -------------------------------
# Plotting the results
plt.figure(figsize=(12, 6))

# Current pattern (solid lines)
plt.plot(current_N, label='Current N', marker='o', color='blue')
plt.plot(current_C, label='Current C', marker='o', color='green')
plt.plot(current_L, label='Current L', marker='o', color='red')

# Full-variance pattern (dashed lines)
plt.plot(full_var_N, '--', label='FullVar N', color='cyan')
plt.plot(full_var_C, '--', label='FullVar C', color='lime')
plt.plot(full_var_L, '--', label='FullVar L', color='orange')

# Labels, title, legend, and grid
plt.xlabel("Turn")
plt.ylabel("Score (0-5)")
plt.title("Comparison: High-Intensity vs Full-Variance Conversation Dynamics (Synthetic Data)")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
