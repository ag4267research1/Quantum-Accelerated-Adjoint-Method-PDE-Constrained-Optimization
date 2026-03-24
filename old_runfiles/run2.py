import yaml
import sys


def main():

    # --------------------------------------------------
    # Check input
    # --------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage: python run.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # Select experiment
    # --------------------------------------------------

    experiment = config.get("experiment", "heat")

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


if __name__ == "__main__":
    main()