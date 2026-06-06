import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

from src.models.heat_model import HeatModel
from src.optimization.optimizer import Optimizer

from src.classical.classical_solver import (
    state_solver,
    adjoint_solver as classical_adjoint_solver,
    inner_product as classical_inner_product,
)
from src.quantum.qlsa_solver import (
    adjoint_solver as qlsa_solver,
    inner_product as swap_test_inner_product,
)
from src.quantum.spectral_gradient import spectral_gradient


# ------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------

def _get_solver_components(mode):
    """Return (adjoint_solver, inner_product, gradient_estimator) for the given mode."""
    if mode == "classical":
        return classical_adjoint_solver, classical_inner_product, None

    if mode == "hybrid":
        def _hybrid_gradient_estimator(**kwargs):
            return spectral_gradient(state_solver=state_solver, **kwargs)
        return qlsa_solver, swap_test_inner_product, _hybrid_gradient_estimator

    raise ValueError(f"Unknown solver mode: {mode}")


def _build_optimize_kwargs(config, mode):
    """Build the kwargs dict forwarded into optimizer.optimize()."""
    optimizer_cfg = config.get("optimizer", {})
    kwargs = {
        "alpha": optimizer_cfg.get("alpha", 1e-3),
        "use_backtracking": optimizer_cfg.get("use_backtracking", True),
        "armijo_c": optimizer_cfg.get("armijo_c", 1e-6),
        "backtracking_tau": optimizer_cfg.get("backtracking_tau", 0.5),
        "min_step": optimizer_cfg.get("min_step", 1e-10),
        "max_backtracks": optimizer_cfg.get("max_backtracks", 30),
    }

    if mode == "hybrid":
        quantum_cfg = config.get("quantum", {})
        kwargs.update({
            "shots": quantum_cfg.get("shots", 64),
            "delta": quantum_cfg.get("delta", 1e-3),
            "N": quantum_cfg.get("spectral_points", 16),
            "backend_mode": quantum_cfg.get("backend_mode", "aer"),
            "ibm_backend_name": quantum_cfg.get("ibm_backend_name", None),
            "ibm_channel": quantum_cfg.get("ibm_channel", None),
            "ibm_token": quantum_cfg.get("ibm_token", None),
            "ibm_instance": quantum_cfg.get("ibm_instance", None),
            "ibm_use_least_busy": quantum_cfg.get("ibm_use_least_busy", False),
            "ionq_token": quantum_cfg.get("ionq_token", None),
            "ionq_backend_name": quantum_cfg.get("ionq_backend_name", None),
        })

    return kwargs


def _build_optimizer(model, mode):
    """Construct an Optimizer wired to classical or hybrid solver components."""
    adj_solver, ip, grad_estimator = _get_solver_components(mode)
    return Optimizer(
        model=model,
        state_solver=state_solver,
        adjoint_solver=adj_solver,
        inner_product=ip,
        control_gradient_estimator=grad_estimator,
    )


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def save_history_plots(history, output_dir, mode):
    """Save per-run objective, gradient norm, and condition number plots."""
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


def save_superimposed_history_plots(histories_by_n, output_dir, mode):
    """Save objective, normalized objective, gradient norm, and condition number
    plots with all state sizes overlaid on the same axes."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    for n, history in histories_by_n.items():
        plt.plot(np.arange(len(history["objective"])), history["objective"], label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(f"Objective vs Iteration ({mode}, fixed $n_x$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_superimposed_objective_vs_iteration.png"))
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        obj = np.asarray(history["objective"], dtype=float)
        obj_norm = obj / obj[0] if len(obj) > 0 and not np.isclose(obj[0], 0.0) else obj
        plt.plot(np.arange(len(obj)), obj_norm, label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel(r"Objective / $J_0$")
    plt.title(f"Normalized Objective vs Iteration ({mode}, fixed $n_x$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_superimposed_objective_normalized_vs_iteration.png"))
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        plt.plot(np.arange(len(history["gradient_norm"])), history["gradient_norm"], label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm vs Iteration ({mode}, fixed $n_x$)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_superimposed_gradient_vs_iteration.png"))
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        plt.plot(np.arange(len(history["condition_number"])), history["condition_number"], label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.title(f"Condition Number vs Iteration ({mode}, fixed $n_x$)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_superimposed_condition_number_vs_iteration.png"))
    plt.close()


def _history_value(history, key, idx, default=np.nan):
    """Safe index into a history list; returns default if out of range."""
    values = history.get(key, [])
    return values[idx] if idx < len(values) else default


def save_superimposed_history_csv(histories_by_n, output_dir, mode, nx):
    """Export per-iteration data for all state sizes to a single CSV."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{mode}_superimposed_iteration_history.csv")

    fieldnames = [
        "mode", "n", "nx", "iteration",
        "objective", "gradient_norm", "condition_number", "step_size",
        "shots_per_iteration", "backtrack_count", "accepted_step",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n, history in histories_by_n.items():
            num_rows = max(
                (len(v) for v in history.values() if isinstance(v, list)),
                default=0,
            )
            for k in range(num_rows):
                writer.writerow({
                    "mode": mode,
                    "n": n,
                    "nx": nx,
                    "iteration": k,
                    "objective": _history_value(history, "objective", k),
                    "gradient_norm": _history_value(history, "gradient_norm", k),
                    "condition_number": _history_value(history, "condition_number", k),
                    "step_size": _history_value(history, "step_size", k),
                    "shots_per_iteration": _history_value(history, "shots_per_iteration", k, 0),
                    "backtrack_count": _history_value(history, "backtrack_count", k, 0),
                    "accepted_step": _history_value(history, "accepted_step", k, 1),
                })

    print(f"Saved superimposed history CSV to: {csv_path}")


# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------

def plot_solution(config):
    """Run the optimizer and plot the final temperature profile."""
    n = config["model"]["n"]
    nx = config["model"]["nx"]
    mode = config["solver"]["mode"]

    model = HeatModel(n=n, nx=nx)
    x0 = 0.01 * np.ones(model.nx)

    optimizer = _build_optimizer(model, mode)
    optimizer_cfg = config.get("optimizer", {})
    optimize_kwargs = _build_optimize_kwargs(config, mode)

    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", "output")

    result = optimizer.optimize(
        x0,
        max_iter=optimizer_cfg.get("max_iter", 100),
        **optimize_kwargs,
    )

    save_history_plots(result.history, output_dir=output_dir, mode=mode)

    u = state_solver(model, result.x_star)

    plt.figure()
    plt.plot(model.grid, u)
    plt.xlabel("y")
    plt.ylabel("Temperature (state u)")
    plt.title(f"Heat Equation Solution ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_solution_profile.png"))
    plt.close()


def scaling_experiment(config):
    """Measure runtime as the number of state variables increases."""
    sizes = config["scaling"]["sizes"]
    iterations = config["scaling"]["iterations"]
    nx = config["model"]["nx"]
    mode = config["solver"]["mode"]

    optimize_kwargs = _build_optimize_kwargs(config, mode)

    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    runtimes = []

    for n in sizes:
        model = HeatModel(n=n, nx=nx)
        x0 = 0.01 * np.ones(model.nx)
        optimizer = _build_optimizer(model, mode)

        start = time.time()
        optimizer.optimize(x0, max_iter=iterations, **optimize_kwargs)
        runtimes.append(time.time() - start)

    plt.figure()
    plt.plot(sizes, runtimes, marker="o")
    plt.xlabel("Number of State Variables")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Scaling ({mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_scaling_runtime.png"))
    plt.close()


def superimposed_history_experiment(config):
    """Run the optimizer for multiple state sizes with fixed nx and
    save superimposed convergence plots and a CSV."""
    sizes = config["scaling"]["sizes"]
    nx = config["model"]["nx"]
    mode = config["solver"]["mode"]

    optimizer_cfg = config.get("optimizer", {})
    optimize_kwargs = _build_optimize_kwargs(config, mode)

    plots_cfg = config.get("plots", {})
    output_dir = plots_cfg.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    histories_by_n = {}

    for n in sizes:
        model = HeatModel(n=n, nx=nx)
        x0 = 0.01 * np.ones(model.nx)
        optimizer = _build_optimizer(model, mode)

        result = optimizer.optimize(
            x0,
            max_iter=optimizer_cfg.get("max_iter", 100),
            **optimize_kwargs,
        )
        histories_by_n[n] = result.history

    save_superimposed_history_plots(histories_by_n, output_dir, mode)
    save_superimposed_history_csv(histories_by_n, output_dir, mode, nx)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def run_experiment(config):
    """Dispatch to the experiments enabled in the config."""
    plots_cfg = config.get("plots", {})

    if plots_cfg["show_solution"]:
        print("Running solution experiment...")
        plot_solution(config)

    if plots_cfg["show_scaling"]:
        print("Running scaling experiment...")
        scaling_experiment(config)

    if plots_cfg.get("show_superimposed_histories", False):
        print("Running superimposed history experiment...")
        superimposed_history_experiment(config)
