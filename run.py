import yaml
import sys

from src.experiments.heat_experiment import run_experiment


def main():

    # check if user provided config file
    if len(sys.argv) < 2:
        print("Usage: python run.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    # load YAML configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()