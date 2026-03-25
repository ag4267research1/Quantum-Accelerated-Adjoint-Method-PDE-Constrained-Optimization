import numpy as np
import matplotlib.pyplot as plt
import time
import os

from src.models.heat_model import HeatModel
from src.optimization.optimizer import Optimizer

# classical components
from src.classical.classical_solver import (
    state_solver,
    adjoint_solver as classical_adjoint_solver,
    inner_product as classical_inner_product
)

# hybrid quantum components
from src.quantum.qlsa_solver import adjoint_solver as qlsa_solver
from src.quantum.swap_test import inner_product as swap_test_inner_product
from src.quantum.spectral_gradient import spectral_gradient


# ------------------------------------------------------------
# Helper function: choose solver components
# ------------------------------------------------------------

def get_solver_components(mode):
    """
    Select the solver components depending on the chosen mode.

    Parameters
    ----------
    mode : str
        "classical" or "hybrid"

    Returns
    -------
    adjoint_solver
        Function used to solve the adjoint equation.

    inner_product
        Function used in the gradient assembly step.

    gradient_estimator
        Function used to estimate the control gradient.
        For the classical mode this is None, so the optimizer
        uses the model's analytic derivative. For the hybrid
        mode this wraps the spectral gradient routine and
        passes the classical state solver into it.
    """

    if mode == "classical":

        return (
            classical_adjoint_solver,
            classical_inner_product,
            None
        )

    elif mode == "hybrid":

        def hybrid_gradient_estimator(**kwargs):
            """
            Wrapper around the spectral gradient estimator.

            The spectral gradient routine needs access to the
            classical state solver in order to evaluate the
            reduced objective at perturbed control values.
            """

            return spectral_gradient(
                state_solver=state_solver,
                **kwargs
            )

        return (
            qlsa_solver,
            swap_test_inner_product,
            hybrid_gradient_estimator
        )

    else:
        raise ValueError(f"Unknown solver mode: {mode}")


# ------------------------------------------------------------
# Solution experiment
# ------------------------------------------------------------

def plot_solution(config):
    """
    Run the solution experiment.

    This experiment:
    1. Builds the heat model
    2. Selects either classical or hybrid components
    3. Runs the optimizer
    4. Solves for the final state
    5. Plots the temperature profile
    """

    n = config["model"]["n"]
    nx = config["model"]["nx"]
    max_iter = config["optimizer"]["max_iter"]

    mode = config["solver"]["mode"]

    model = HeatModel(n=n, nx=nx)

    # small initial control
    x0 = 0.01 * np.ones(model.nx)

    # choose solver components
    adjoint_solver, inner_product, gradient_estimator = get_solver_components(mode)

    optimizer = Optimizer(
        model=model,
        state_solver=state_solver,
        adjoint_solver=adjoint_solver,
        inner_product=inner_product,
        control_gradient_estimator=gradient_estimator
    )

    # run optimization
    result = optimizer.optimize(x0, max_iter=max_iter)

    # optimizer returns the optimal control
    x = result.x_star

    # compute final state from the optimal control
    u = state_solver(model, x)

    grid = model.grid

    save_dir = config.get("output", {}).get("save_dir", "plots")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(grid, u)
    plt.xlabel("y")
    plt.ylabel("Temperature (state u)")
    plt.title(f"Heat Equation Solution ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heat_solution_{mode}.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# Scaling experiment
# ------------------------------------------------------------

def scaling_experiment(config):
    """
    Run the scaling experiment.

    This experiment measures runtime as the number of
    state variables increases.
    """

    sizes = config["scaling"]["sizes"]
    iterations = config["scaling"]["iterations"]
    nx = config["model"]["nx"]

    mode = config["solver"]["mode"]

    # choose solver components once
    adjoint_solver, inner_product, gradient_estimator = get_solver_components(mode)

    runtimes = []

    for n in sizes:

        model = HeatModel(n=n, nx=nx)

        # small initial control
        x0 = 0.01 * np.ones(model.nx)

        optimizer = Optimizer(
            model=model,
            state_solver=state_solver,
            adjoint_solver=adjoint_solver,
            inner_product=inner_product,
            control_gradient_estimator=gradient_estimator
        )

        start = time.time()
        optimizer.optimize(x0, max_iter=iterations)
        end = time.time()

        runtimes.append(end - start)

    save_dir = config.get("output", {}).get("save_dir", "plots")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(sizes, runtimes, marker="o")
    plt.xlabel("Number of State Variables")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Scaling ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heat_scaling_{mode}.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------

def run_experiment(config):
    """
    Run the experiments selected in the configuration file.
    """

    if config["plots"]["show_solution"]:
        print("Running solution experiment...")
        plot_solution(config)

    if config["plots"]["show_scaling"]:
        print("Running scaling experiment...")
        scaling_experiment(config)