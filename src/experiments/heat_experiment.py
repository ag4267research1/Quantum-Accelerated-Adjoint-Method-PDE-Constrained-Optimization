import os
import numpy as np
import matplotlib.pyplot as plt
import time

from src.models.heat_model import HeatModel
from src.optimization.optimizer import Optimizer

# classical components
from src.classical.classical_solver import (
    state_solver,
    adjoint_solver as classical_adjoint_solver,
    inner_product as classical_inner_product
)

# hybrid quantum components
from src.quantum.qlsa_solver import (
    adjoint_solver as qlsa_solver,
    inner_product as swap_test_inner_product,
)
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
            hybrid_gradient_estimator  # CHANGED: this was None
        )

    else:
        raise ValueError(f"Unknown solver mode: {mode}")


# ------------------------------------------------------------
# Helper function: save optimization history plots
# ------------------------------------------------------------

def save_history_plots(history, output_dir, mode):
    """
    Save iteration vs objective, gradient norm, and condition number plots.
    """

    os.makedirs(output_dir, exist_ok=True)

    it_obj = np.arange(len(history["objective"]))
    it_grad = np.arange(len(history["gradient_norm"]))
    it_cond = np.arange(len(history["condition_number"]))

    plt.figure()
    plt.plot(it_obj, history["objective"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(f"Objective vs Iteration ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_objective_vs_iteration.png"))
    plt.close()

    plt.figure()
    plt.plot(it_grad, history["gradient_norm"])
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm vs Iteration ({mode})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_gradient_vs_iteration.png"))
    plt.close()

    plt.figure()
    plt.plot(it_cond, history["condition_number"])
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.title(f"Condition Number vs Iteration ({mode})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_condition_number_vs_iteration.png"))
    plt.close()


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

    optimize_kwargs = {}

    # forward optimizer config to optimizer.optimize(...)
    optimizer_cfg = config.get("optimizer", {})
    optimize_kwargs["alpha"] = optimizer_cfg.get("alpha", 1e-3)
    optimize_kwargs["use_backtracking"] = optimizer_cfg.get("use_backtracking", True)
    optimize_kwargs["armijo_c"] = optimizer_cfg.get("armijo_c", 1e-6)
    optimize_kwargs["backtracking_tau"] = optimizer_cfg.get("backtracking_tau", 0.5)
    optimize_kwargs["min_step"] = optimizer_cfg.get("min_step", 1e-10)
    optimize_kwargs["max_backtracks"] = optimizer_cfg.get("max_backtracks", 30)

    # CHANGED: forward surrogate controls too
    optimize_kwargs["use_quantum_surrogate"] = optimizer_cfg.get("use_quantum_surrogate", False)
    optimize_kwargs["quantum_beta"] = optimizer_cfg.get("quantum_beta", 0.05)

    if mode == "hybrid":
        quantum_cfg = config.get("quantum", {})
        optimize_kwargs["shots"] = quantum_cfg.get("shots", 64)
        optimize_kwargs["delta"] = quantum_cfg.get("delta", 1e-3)
        optimize_kwargs["N"] = quantum_cfg.get("spectral_points", 16)
        optimize_kwargs["use_preconditioning"] = quantum_cfg.get("use_preconditioning", True)

    # define one output directory and save all plots there
    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", "output")

    # run optimization
    result = optimizer.optimize(x0, max_iter=max_iter, **optimize_kwargs)

    # save history plots in the output directory
    save_history_plots(result.history, output_dir=output_dir, mode=mode)

    # optimizer returns the optimal control
    x = result.x_star

    # compute final state from the optimal control
    u = state_solver(model, x)

    grid = model.grid

    plt.figure()
    plt.plot(grid, u)
    plt.xlabel("y")
    plt.ylabel("Temperature (state u)")
    plt.title(f"Heat Equation Solution ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_solution_profile.png"))
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

    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

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

        optimize_kwargs = {}

        # forward optimizer config to optimizer.optimize(...)
        optimizer_cfg = config.get("optimizer", {})
        optimize_kwargs["alpha"] = optimizer_cfg.get("alpha", 1e-3)
        optimize_kwargs["use_backtracking"] = optimizer_cfg.get("use_backtracking", True)
        optimize_kwargs["armijo_c"] = optimizer_cfg.get("armijo_c", 1e-6)
        optimize_kwargs["backtracking_tau"] = optimizer_cfg.get("backtracking_tau", 0.5)
        optimize_kwargs["min_step"] = optimizer_cfg.get("min_step", 1e-10)
        optimize_kwargs["max_backtracks"] = optimizer_cfg.get("max_backtracks", 30)

        # CHANGED: forward surrogate controls too
        optimize_kwargs["use_quantum_surrogate"] = optimizer_cfg.get("use_quantum_surrogate", False)
        optimize_kwargs["quantum_beta"] = optimizer_cfg.get("quantum_beta", 0.05)

        if mode == "hybrid":
            quantum_cfg = config.get("quantum", {})
            optimize_kwargs["shots"] = quantum_cfg.get("shots", 64)
            optimize_kwargs["delta"] = quantum_cfg.get("delta", 1e-3)
            optimize_kwargs["N"] = quantum_cfg.get("spectral_points", 16)
            optimize_kwargs["use_preconditioning"] = quantum_cfg.get("use_preconditioning", True)

        start = time.time()

        optimizer.optimize(x0, max_iter=iterations, **optimize_kwargs)

        end = time.time()

        runtimes.append(end - start)

    plt.figure()
    plt.plot(sizes, runtimes, marker="o")
    plt.xlabel("Number of State Variables")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Scaling ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_scaling_runtime.png"))
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