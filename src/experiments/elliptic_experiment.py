"""
Elliptic PDE Experiment

Generates:
1. Runtime vs number of state variables
2. Condition number vs number of state variables
3. Final solution plot
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from src.models.elliptic_model import EllipticModel
from src.optimization.optimizer import Optimizer


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_experiment(config):
    """
    Runs full experiment suite and generates plots.
    """

    sizes = config.get("sizes", [8, 16, 24, 32])
    runtimes = []
    condition_numbers = []

    last_result = None
    last_model = None
    last_state_solver = None

    for n in sizes:

        print(f"\n Number of state variables: {n}")

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
        u0 = np.zeros(model.num_dofs)

        # --------------------------------------------
        # Build system once (for condition number)
        # --------------------------------------------

        A, b0 = model.build_system(u0)

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

        solver_type = config.get("solver", "classical")

        if solver_type == "classical":
            from src.classical.classical_solver import state_solver
            from src.classical.classical_solver import adjoint_solver
            from src.classical.classical_solver import inner_product

        elif solver_type == "hybrid":
            # state solve is still classical in the hybrid workflow
            from src.classical.classical_solver import state_solver
            from src.quantum.qlsa_solver import adjoint_solver
            from src.quantum.swap_test import inner_product

        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        last_state_solver = state_solver

        optimizer = Optimizer(
            model=model,
            state_solver=state_solver,
            adjoint_solver=adjoint_solver,
            inner_product=inner_product
        )

        start = time.time()
        result = optimizer.optimize(
            u0,
            max_iter=config.get("max_iter", 10),
            alpha=config.get("step_size", 1e-2)
        )
        end = time.time()

        runtime = end - start
        runtimes.append(runtime)

        print(f"Runtime: {runtime:.4f}s | cond(A): {cond_A:.2e}")

        last_result = result

    # ============================================================
    # PLOTS
    # ============================================================

    plot_runtime(sizes, runtimes)
    plot_condition_number(sizes, condition_numbers)
    plot_solution(last_result, last_model, last_state_solver)


# ============================================================
# PLOT 1: Runtime vs state variables
# ============================================================

def plot_runtime(sizes, runtimes):
    """
    Plot: number of state variables vs runtime
    """

    state_vars = sizes

    plt.figure()
    plt.plot(state_vars, runtimes, marker='o')
    plt.xlabel("Number of State Variables")
    plt.ylabel("Runtime (seconds)")
    plt.title("State Variables vs Runtime")
    plt.grid(True)
    plt.show()


# ============================================================
# PLOT 2: Condition number vs state variables
# ============================================================

def plot_condition_number(sizes, conds):
    """
    Plot: number of state variables vs condition number
    """

    state_vars = sizes

    plt.figure()
    plt.plot(state_vars, conds, marker='o')
    plt.xlabel("Number of State Variables")
    plt.ylabel("Condition Number")
    plt.title("State Variables vs Condition Number")
    plt.yscale("log")
    plt.grid(True)
    plt.show()


# ============================================================
# PLOT 3: Solution
# ============================================================

def plot_solution(result, model, state_solver):
    """
    Plot final state y
    """

    x_opt = result.x_star
    y = state_solver(model=model, x=x_opt)

    plt.figure()
    plt.plot(model.x, y, marker='o')
    plt.title("Elliptic PDE Solution")
    plt.xlabel("x")
    plt.ylabel("State y")
    plt.grid(True)
    plt.show()