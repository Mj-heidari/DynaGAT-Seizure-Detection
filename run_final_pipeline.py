"""
DynaGAT One Click Final Pipeline

Flow:
1. Generate cache from CHB-MIT BIDS dataset
2. Run final training
3. Export paper figures

Before running:
- Set CHB-MIT BIDS path in config.py
"""

import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(cmd))

print("="*70)
print("DynaGAT FINAL PIPELINE")
print("="*70)


# Step 1: preprocessing
if Path("run_preprocessing.py").exists():
    run([sys.executable, "run_preprocessing.py"])
else:
    print("Preprocessing script not found. Using existing cache.")

# Step 2: training
run([sys.executable, "run_training.py"])

# Step 3: figures
if Path("generate_all_figures.py").exists():
    run([sys.executable, "generate_all_figures.py"])
else:
    print("Figure generator not found.")

print("\nFINAL PIPELINE COMPLETE")
