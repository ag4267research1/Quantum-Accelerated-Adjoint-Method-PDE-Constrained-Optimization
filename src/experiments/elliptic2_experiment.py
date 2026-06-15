
"""
Elliptic2 PDE Experiment

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
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.models.elliptic2_model import Elliptic2Model
from src.optimization.optimizer import Optimizer
from src.quantum.qlsa_solver import (
    adjoint_solver as qlsa_solver,
    inner_product as swap_test_inner_product,
    clear_quantum_log,
    get_quantum_log,
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
        optimize_kwargs["backend_mode"] = quantum_cfg.get("backend_mode", "aer")
        optimize_kwargs["ibm_backend_name"] = quantum_cfg.get("ibm_backend_name", None)
        optimize_kwargs["ibm_channel"] = quantum_cfg.get("ibm_channel", None)
        optimize_kwargs["ibm_token"] = quantum_cfg.get("ibm_token", None)
        optimize_kwargs["ibm_instance"] = quantum_cfg.get("ibm_instance", None)
        optimize_kwargs["ibm_use_least_busy"] = quantum_cfg.get("ibm_use_least_busy", False)
        optimize_kwargs["ionq_token"] = quantum_cfg.get("ionq_token", None)
        optimize_kwargs["ionq_backend_name"] = quantum_cfg.get("ionq_backend_name", None)
        optimize_kwargs["boost_qpe_resolution"] = quantum_cfg.get("boost_qpe_resolution", False)
        optimize_kwargs["extra_qpe_qubits"] = quantum_cfg.get("extra_qpe_qubits", 2)

    return optimize_kwargs


# ============================================================
# Helper: save optimization history CSVs
# ============================================================

def _history_value(history, key, idx, default=np.nan):
    """Safe index into a history list; returns default if out of range."""
    values = history.get(key, [])
    return values[idx] if idx < len(values) else default


def _aggregate_quantum_log(log, iteration, nx):
    """Return per-iteration quantum metrics aggregated over nx calls."""
    calls = log[iteration * nx : (iteration + 1) * nx]
    if not calls:
        return {"shots": np.nan, "successful_shots": np.nan, "avg_success_rate": np.nan, "avg_residual": np.nan}
    return {
        "shots": calls[0]["shots"],
        "successful_shots": sum(c["successful_shots"] for c in calls if c["successful_shots"] is not None),
        "avg_success_rate": float(np.mean([c["success_rate"] for c in calls if c["success_rate"] is not None])),
        "avg_residual": float(np.mean([c["residual"] for c in calls if c["residual"] is not None])),
    }


def save_iteration_history_csv(histories_by_n, output_dir, exp_name, mode, nx_by_n, quantum_logs_by_n=None):
    """Export per-iteration data for all state sizes to a single CSV."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{exp_name}_{mode}_iteration_history.csv")

    fieldnames = [
        "mode", "n", "nx", "iteration",
        "objective", "gradient_norm", "condition_number", "step_size",
        "shots_per_iteration", "successful_shots", "avg_success_rate", "avg_residual",
        "backtrack_count", "accepted_step",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n, history in histories_by_n.items():
            nx = nx_by_n[n]
            num_rows = max(
                (len(v) for v in history.values() if isinstance(v, list)),
                default=0,
            )
            q_log = (quantum_logs_by_n or {}).get(n, [])
            for k in range(num_rows):
                q = _aggregate_quantum_log(q_log, k, nx) if q_log else {
                    "shots": np.nan, "successful_shots": np.nan,
                    "avg_success_rate": np.nan, "avg_residual": np.nan,
                }
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
                    "successful_shots": q["successful_shots"],
                    "avg_success_rate": q["avg_success_rate"],
                    "avg_residual": q["avg_residual"],
                    "backtrack_count": _history_value(history, "backtrack_count", k, 0),
                    "accepted_step": _history_value(history, "accepted_step", k, 1),
                })

    print(f"Saved iteration history CSV to: {csv_path}")


def _save_quantum_call_log(quantum_logs_by_n, output_dir, exp_name, mode, nx_by_n):
    """Save the raw per-call quantum log to CSV (one row per inner_product call)."""
    csv_path = os.path.join(output_dir, f"{exp_name}_{mode}_quantum_calls.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "nx", "iteration", "call_within_iteration", "shots", "successful_shots", "success_rate", "residual"])
        writer.writeheader()
        for n, log in quantum_logs_by_n.items():
            nx = nx_by_n[n]
            for entry in log:
                iteration = entry["call"] // nx
                call_within = entry["call"] % nx
                writer.writerow({
                    "n": n,
                    "nx": nx,
                    "iteration": iteration,
                    "call_within_iteration": call_within,
                    "shots": entry["shots"],
                    "successful_shots": entry["successful_shots"],
                    "success_rate": entry["success_rate"],
                    "residual": entry["residual"],
                })
    print(f"Saved per-call quantum log to: {csv_path}")


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

    obj = np.asarray(history["objective"], dtype=float)
    grad = np.asarray(history["gradient_norm"], dtype=float)
    cond = np.asarray(history["condition_number"], dtype=float)

    eps = 1e-16

    plt.figure()
    plt.plot(it_obj, np.maximum(obj, eps))
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(f"Objective vs Iteration ({mode}, n={n})")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_objective_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(it_grad, np.maximum(grad, eps))
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm vs Iteration ({mode}, n={n})")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_gradient_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(it_cond, np.maximum(cond, eps))
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.title(f"Condition Number vs Iteration ({mode}, n={n})")
    plt.yscale("log")
    plt.grid(True)

    positive_cond = cond[cond > 0]
    if len(positive_cond) > 0:
        cmin = np.min(positive_cond)
        cmax = np.max(positive_cond)

        if np.isclose(cmin, cmax):
            plt.ylim(0.9 * cmin, 1.1 * cmax)
        else:
            plt.ylim(0.95 * cmin, 1.05 * cmax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_n{n}_condition_number_vs_iteration.png"), dpi=300)
    plt.close()


def save_superimposed_history_plots(histories_by_n, output_dir, exp_name, mode):
    """Save objective, normalized objective, gradient norm, and condition number
    plots with all state sizes overlaid on the same axes (mirrors heat_experiment)."""
    os.makedirs(output_dir, exist_ok=True)

    eps = 1e-16

    plt.figure()
    for n, history in histories_by_n.items():
        obj = np.maximum(np.asarray(history["objective"], dtype=float), eps)
        plt.plot(np.arange(len(obj)), obj, label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(f"Objective vs Iteration ({mode})")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_superimposed_objective_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        obj = np.asarray(history["objective"], dtype=float)
        obj_norm = obj / obj[0] if len(obj) > 0 and not np.isclose(obj[0], 0.0) else obj
        plt.plot(np.arange(len(obj)), obj_norm, label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel(r"Objective / $J_0$")
    plt.title(f"Normalized Objective vs Iteration ({mode})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_superimposed_objective_normalized_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        grad = np.maximum(np.asarray(history["gradient_norm"], dtype=float), eps)
        plt.plot(np.arange(len(grad)), grad, label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm vs Iteration ({mode})")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_superimposed_gradient_vs_iteration.png"), dpi=300)
    plt.close()

    plt.figure()
    for n, history in histories_by_n.items():
        cond = np.maximum(np.asarray(history["condition_number"], dtype=float), eps)
        plt.plot(np.arange(len(cond)), cond, label=f"n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.title(f"Condition Number vs Iteration ({mode})")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name}_{mode}_superimposed_condition_number_vs_iteration.png"), dpi=300)
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

    histories_by_n = {}
    quantum_logs_by_n = {}
    nx_by_n = {}

    mode = _get_mode(config)
    output_dir = _get_output_dir(config)
    exp_name = _get_experiment_name(config)

    plots_cfg = config.get("plots", {})
    show_solution = plots_cfg.get("show_solution", True)
    show_scaling = plots_cfg.get("show_scaling", True)
    show_superimposed_histories = plots_cfg.get("show_superimposed_histories", False)

    for n in sizes:

        print(f"\n Number of state variables: {n} ")

        # --------------------------------------------
        # Update config for this size
        # --------------------------------------------
        config["grid_size"] = n

        # --------------------------------------------
        # Build model
        # --------------------------------------------
        model = Elliptic2Model(config)
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

            # use the merged single-file quantum API already imported above
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

        if mode == "hybrid":
            clear_quantum_log()

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

        if not show_superimposed_histories:
            save_history_plots(result.history, output_dir, exp_name, mode, n)

        histories_by_n[n] = result.history
        nx_by_n[n] = model.num_dofs
        if mode == "hybrid":
            quantum_logs_by_n[n] = get_quantum_log()

        last_result = result

    # ============================================================
    # ITERATION HISTORY CSVs
    # ============================================================

    if show_superimposed_histories:
        save_superimposed_history_plots(histories_by_n, output_dir, exp_name, mode)
        save_iteration_history_csv(
            histories_by_n, output_dir, exp_name, mode, nx_by_n,
            quantum_logs_by_n=quantum_logs_by_n if mode == "hybrid" else None,
        )
        if mode == "hybrid" and quantum_logs_by_n:
            _save_quantum_call_log(quantum_logs_by_n, output_dir, exp_name, mode, nx_by_n)

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
    plt.yscale("log")
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
    plt.yscale("log")
    plt.plot(state_vars, conds, marker='o')
    plt.xlabel("Number of State Variables")
    plt.ylabel("Condition Number")
    plt.title("State Variables vs Condition Number")
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
    plt.title("Elliptic2 PDE Solution")
    plt.xlabel("x")
    plt.ylabel("State u(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_type}_solution.png"), dpi=300)
    plt.close()
