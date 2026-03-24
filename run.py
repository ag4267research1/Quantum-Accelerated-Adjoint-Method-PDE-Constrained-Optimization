"""
Unified experiment runner

This script loads a YAML config file and dispatches
to the correct experiment (heat or elliptic).

Usage:
    python run_experiment.py <config_file>
"""

import yaml
import sys


def main():

    # --------------------------------------------------
    # Check input
    # --------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # Select experiment type
    # --------------------------------------------------

    experiment = config.get("experiment", "heat")

    print(f"\n[INFO] Running experiment: {experiment}")

    # --------------------------------------------------
    # Dispatch to correct experiment
    # --------------------------------------------------

    if experiment == "heat":
        from src.experiments.heat_experiment import run_experiment

    elif experiment == "elliptic":
        from src.experiments.elliptic_experiment import run_experiment

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # --------------------------------------------------
    # Run experiment
    # --------------------------------------------------

    run_experiment(config)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()