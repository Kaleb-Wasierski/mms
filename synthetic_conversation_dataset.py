"""
synthetic_conversation_dataset.py
---------------------------------
Generates a synthetic dataset of conversation dynamics for demonstration purposes.
Features per turn:
- N: Novelty (0-5)
- C: Constraint Pressure (0-5)
- L: Cognitive Load (0-5)
- DeltaC: Change in constraint from previous turn
- DeltaL: Change in load from previous turn
All data is synthetic; no personal information is used.
"""

import numpy as np
import pandas as pd

# Number of turns to simulate
turns = 40
np.random.seed(42)  # reproducibility

# -------------------------------
# Generate Current High-Intensity Pattern (plateau effect)
current_N = np.random.randint(3, 6, size=turns)
current_C = np.random.randint(3, 6, size=turns)
current_L = np.random.randint(3, 6, size=turns)

# Calculate deltas
DeltaC = np.diff(current_C, prepend=current_C[0])
DeltaL = np.diff(current_L, prepend=current_L[0])

# -------------------------------
# Create DataFrame
df = pd.DataFrame({
    "Turn": np.arange(1, turns + 1),
    "N": current_N,
    "C": current_C,
    "L": current_L,
    "DeltaC": DeltaC,
    "DeltaL": DeltaL
})

# -------------------------------
# Save to CSV
df.to_csv("synthetic_conversation_data.csv", index=False)

print("Synthetic conversation dataset generated and saved as 'synthetic_conversation_data.csv'")
print(df)
