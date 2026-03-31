"""
Elliptic PDE Experiment

Generates:
1. Runtime vs number of state variables
2. Condition number vs number of state variables
3. Final solution plot
4. Objective vs iteration
5. Gradient norm vs iteration
6. Condition number vs iteration
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt

from src.models.elliptic_model import EllipticModel
from src.optimization.optimizer import Optimizer
from src.quantum.qlsa_solver import (
    adjoint_solver as qlsa_solver,
    inner_product as swap_test_inner_product,
)


# ============================================================
# Helper: config access
# ============================================================

def _get_mode(config):
    """
    Read solver mode from either
        config["solver"]["mode"]
    or the older flat style
        config["solver"].
    """
    solver_cfg = config.get("solver", "classical")
    if isinstance(solver_cfg, dict):
        return solver_cfg.get("mode", "classical")
    return solver_cfg


def _get_output_dir(config):
    """
    Read output directory from either
        config["plots"]["output_dir"]
    or the older flat style
        config["output_dir"].
    """
    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", config.get("output_dir", "plots"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _get_experiment_name(config):
    """
    Read experiment name for filenames.
    """
    return config.get("experiment", config.get("experiment_type", "exp"))


def _build_optimize_kwargs(config, mode):
    """
    Build kwargs forwarded into optimizer.optimize(...).

    This keeps compatibility with both the newer nested YAML style and the
    older flat config style.
    """
    optimize_kwargs = {}

    optimizer_cfg = config.get("optimizer", {})
    optimize_kwargs["alpha"] = optimizer_cfg.get("alpha", config.get("step_size", 1e-2))
    optimize_kwargs["use_backtracking"] = optimizer_cfg.get("use_backtracking", True)
    optimize_kwargs["armijo_c"] = optimizer_cfg.get("armijo_c", 1e-6)
    optimize_kwargs["backtracking_tau"] = optimizer_cfg.get("backtracking_tau", 0.5)
    optimize_kwargs["min_step"] = optimizer_cfg.get("min_step", 1e-10)
    optimize_kwargs["max_backtracks"] = optimizer_cfg.get("max_backtracks", 30)

    if mode == "hybrid":
        quantum_cfg = config.get("quantum", {})
        optimize_kwargs["shots"] = quantum_cfg.get("shots", 64)
        optimize_kwargs["delta"] = quantum_cfg.get("delta", 1e-3)
        optimize_kwargs["N"] = quantum_cfg.get("spectral_points", 16)
        optimize_kwargs["use_preconditioning"] = quantum_cfg.get("use_preconditioning", False)

    return optimize_kwargs


# ============================================================
# Helper: save optimization history plots
# ============================================================

def save_history_plots(history, output_dir, exp_name, mode, n):
    """
    Save the same history plots as the heat experiment:
    1. objective vs iteration
    2. gradient norm vs iteration
    3. condition number vs iteration
    """

    os.makedirs(output_dir, exist_ok=True)

    it_obj = np.arange(len(history["objective"]))
    it_grad = np.arange(len(history["gradient_norm"]))
    it_cond = np.arange(len(history["condition_number"]))

    plt.figure()
    plt.plot(it_obj, history["objective"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(f"Objective vs Iteration ({mode}, n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_objective_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(it_grad, history["gradient_norm"])
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm vs Iteration ({mode}, n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_gradient_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(it_cond, history["condition_number"])
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.title(f"Condition Number vs Iteration ({mode}, n={n})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_condition_number_vs_iteration.png"), dpi=300)
    plt.close()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_experiment(config):
    """
    Runs full experiment suite and generates plots.
    """

    scaling_cfg = config.get("scaling", {})
    sizes = scaling_cfg.get("sizes", config.get("sizes", [8, 16, 24, 32]))
    runtimes = []
    condition_numbers = []

    last_result = None
    last_model = None
    last_state_solver = None

    mode = _get_mode(config)
    output_dir = _get_output_dir(config)
    exp_name = _get_experiment_name(config)

    plots_cfg = config.get("plots", {})
    show_solution = plots_cfg.get("show_solution", True)
    show_scaling = plots_cfg.get("show_scaling", True)

    for n in sizes:

        print(f"\n Number of state variables: {n} ")

        # --------------------------------------------
        # Update config for this size
        # --------------------------------------------
        config["grid_size"] = n

        # --------------------------------------------
        # Build model
        # --------------------------------------------
        model = EllipticModel(config)
        last_model = model

        # initial control
        x0 = np.ones(model.num_dofs)

        # --------------------------------------------
        # Build system once (for condition number)
        # --------------------------------------------
        A, b0 = model.build_system(x0)

        print(f"experiment_type = {model.exp_type}")
        print(f"A[0,0] = {A[0,0]:.6e}")
        if A.shape[1] > 1:
            print(f"A[0,1] = {A[0,1]:.6e}")
        print(f"||b|| = {np.linalg.norm(b0):.6e}")

        cond_A = np.linalg.cond(A)
        condition_numbers.append(cond_A)

        # --------------------------------------------
        # Run optimizer (measure time)
        # --------------------------------------------
        if mode == "classical":
            from src.classical.classical_solver import state_solver
            from src.classical.classical_solver import adjoint_solver
            from src.classical.classical_solver import inner_product

        elif mode == "hybrid":
            # state solve is still classical in the hybrid workflow
            from src.classical.classical_solver import state_solver

            # CHANGED: use the merged single-file quantum API already imported above
            adjoint_solver = qlsa_solver
            inner_product = swap_test_inner_product

        else:
            raise ValueError(f"Unknown solver mode: {mode}")

        last_state_solver = state_solver

        optimizer = Optimizer(
            model=model,
            state_solver=state_solver,
            adjoint_solver=adjoint_solver,
            inner_product=inner_product
        )

        optimize_kwargs = _build_optimize_kwargs(config, mode)

        optimizer_cfg = config.get("optimizer", {})
        max_iter = optimizer_cfg.get("max_iter", config.get("max_iter", 10))

        start = time.time()
        result = optimizer.optimize(
            x0,
            max_iter=max_iter,
            **optimize_kwargs
        )
        end = time.time()

        runtime = end - start
        runtimes.append(runtime)

        print(f"Runtime: {runtime:.4f}s | cond(A): {cond_A:.2e}")

        save_history_plots(result.history, output_dir, exp_name, mode, n)

        last_result = result

    # ============================================================
    # PLOTS
    # ============================================================

    if show_scaling:
        plot_runtime(sizes, runtimes, config)
        plot_condition_number(sizes, condition_numbers, config)

    if show_solution:
        plot_solution(last_result, last_model, last_state_solver, config)


# ============================================================
# Helper: output directory
# ============================================================

def get_output_dir(config):
    """
    Get output directory for saved plots.
    """
    return _get_output_dir(config)


# ============================================================
# PLOT 1: Runtime vs state variables
# ============================================================

def plot_runtime(sizes, runtimes, config):
    """
    Plot: number of state variables vs runtime
    """

    state_vars = sizes
    output_dir = get_output_dir(config)
    exp_type = _get_experiment_name(config)

    plt.figure()
    plt.plot(state_vars, runtimes, marker='o')
    plt.xlabel("Number of State Variables")
    plt.ylabel("Runtime (seconds)")
    plt.title("State Variables vs Runtime")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_type}_runtime.png"), dpi=300)
    plt.close()


# ============================================================
# PLOT 2: Condition number vs state variables
# ============================================================

def plot_condition_number(sizes, conds, config):
    """
    Plot: number of state variables vs condition number
    """

    state_vars = sizes
    output_dir = get_output_dir(config)
    exp_type = _get_experiment_name(config)

    plt.figure()
    plt.plot(state_vars, conds, marker='o')
    plt.xlabel("Number of State Variables")
    plt.ylabel("Condition Number")
    plt.title("State Variables vs Condition Number")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_type}_condition_number.png"), dpi=300)
    plt.close()


# ============================================================
# PLOT 3: Solution
# ============================================================

def plot_solution(result, model, state_solver, config):
    """
    Plot final state u(x)
    """

    x_opt = result.x_star
    u = state_solver(model=model, x=x_opt)

    output_dir = get_output_dir(config)
    exp_type = _get_experiment_name(config)

    plt.figure()
    plt.plot(model.x, u, marker='o')
    plt.title("Elliptic PDE Solution")
    plt.xlabel("x")
    plt.ylabel("State u(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_type}_solution.png"), dpi=300)
    plt.close()