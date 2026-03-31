from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


Array = np.ndarray


@dataclass
class OptimizationResult:
    x_star: Array
    objective_value: float
    gradient_norm: float
    iterations: int
    converged: bool
    history: Dict[str, list]


class Optimizer:
    """
    Generic optimizer for PDE-constrained optimization.

    This class is designed so that the optimizer itself does not care whether
    the subroutines are classical or quantum. It only calls the functions
    provided to it.
    """

    def __init__(
        self,
        model: Any,
        state_solver: Callable[..., Array],
        adjoint_solver: Callable[..., Any],
        inner_product: Callable[..., float],
        control_gradient_estimator: Optional[Callable[..., Array]] = None,
        line_search: Optional[Callable[..., float]] = None,
    ) -> None:
        self.model = model
        self.state_solver = state_solver
        self.adjoint_solver = adjoint_solver
        self.inner_product = inner_product
        self.control_gradient_estimator = control_gradient_estimator
        self.line_search = line_search

    def optimize(
        self,
        x0: Array,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 1e-3,
        store_history: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run gradient-based optimization.

        Parameters
        ----------
        x0 : np.ndarray
            Initial control.

        max_iter : int
            Maximum number of optimization iterations.

        tol : float
            Stopping tolerance on gradient norm.

        alpha : float
            Default step size if no line search is supplied.

        store_history : bool
            Whether to store objective / gradient history.

        verbose : bool
            Whether to print progress.

        kwargs : Any
            Extra arguments forwarded to subroutines.

        Returns
        -------
        OptimizationResult
        """

        x = np.array(x0, dtype=float).copy()

        history = {
            "objective": [],
            "gradient_norm": [],
            "step_size": [],
            "condition_number": [],
        }

        converged = False
        J = np.nan
        grad = np.zeros_like(x)
        grad_norm = np.inf

        # CHANGED: normalize user/YAML inputs so formatting/math does not fail
        alpha = float(alpha)

        # CHANGED: allow backtracking to be controlled from kwargs / YAML
        use_backtracking = kwargs.get("use_backtracking", True)
        if isinstance(use_backtracking, str):
            use_backtracking = use_backtracking.strip().lower() in ("true", "1", "yes", "y", "on")

        k = 0
        while k < max_iter and grad_norm >= 1e-3:

            # -------------------------------------------------
            # Step 1: Solve state equation c(u,x)=0
            # -------------------------------------------------
            u = self.state_solver(model=self.model, x=x, **kwargs)

            # -------------------------------------------------
            # Step 2: Evaluate objective
            # -------------------------------------------------
            J = float(self.model.objective(u, x))

            # -------------------------------------------------
            # Step 3: Assemble Jacobian and state derivative
            # -------------------------------------------------
            A = self.model.jacobian(u, x)
            g_u = self.model.dJ_du(u, x)

            try:
                cond_A = float(np.linalg.cond(A))
            except Exception:
                cond_A = np.inf

            # -------------------------------------------------
            # Step 3b: Control gradient term
            # Classical default: exact / analytic dJ_dx
            # Quantum later: spectral gradient estimator
            # -------------------------------------------------
            if self.control_gradient_estimator is None:
                g_x = self.model.dJ_dx(u, x)
            else:
                g_x = self.control_gradient_estimator(
                    model=self.model,
                    u=u,
                    x=x,
                    **kwargs,
                )

            # -------------------------------------------------
            # Step 4: Solve adjoint equation A^T p = g_u
            # Classical: returns vector p
            # Quantum: may return a quantum state, a quantum state and scaling,
            # or a quantum state with diagnostics
            # -------------------------------------------------
            adjoint_output = self.adjoint_solver(
                A=A,
                rhs=g_u,
                model=self.model,
                u=u,
                x=x,
                **kwargs,
            )

            # Classical solver returns p
            # Quantum solver may return:
            #   - p_state
            #   - (p_state, p_scale)
            #   - (p_state, diagnostics)
            if isinstance(adjoint_output, tuple):
                p_state = adjoint_output[0]

                # If it is a numeric scale, use it.
                # If it is diagnostics (e.g. a dict), ignore it here and use scale 1.
                if len(adjoint_output) > 1 and np.isscalar(adjoint_output[1]):
                    p_scale = float(adjoint_output[1])
                else:
                    p_scale = 1.0
            else:
                p_state = adjoint_output
                p_scale = 1.0

            # -------------------------------------------------
            # Step 5: Assemble reduced gradient
            # grad_i = dJ/dx_i - <p, dc/dx_i>
            # -------------------------------------------------
            grad = np.zeros_like(x, dtype=float)

            for i in range(len(x)):

                # derivative of constraint wrt control variable
                w_i = self.model.dc_dx_i(u, x, i)

                # compute <p , dc/dx_i>
                z_i = self.inner_product(
                    left=p_state,
                    right=w_i,
                    model=self.model,
                    u=u,
                    x=x,
                    i=i,
                    **kwargs,
                )

                # reduced gradient component
                grad[i] = g_x[i] - p_scale * z_i

            grad_norm = float(np.linalg.norm(grad))

            # stop immediately if the gradient becomes non-finite.
            if not np.isfinite(grad_norm) or not np.all(np.isfinite(grad)):
                if verbose:
                    print("Stopping: non-finite gradient detected.")
                break

            if store_history:
                history["objective"].append(J)
                history["gradient_norm"].append(grad_norm)
                history["condition_number"].append(cond_A)

            if verbose:
                print(f"iter={k:04d} | J={J:.6e} | ||grad||={grad_norm:.12e}")

            # -------------------------------------------------
            # Step 6: Stopping test
            # -------------------------------------------------
            if grad_norm <= tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {k}.")
                break

            if grad_norm < 1e-3:
                converged = True
                if verbose:
                    print(f"Converged at iteration {k} (gradient norm below 1e-3).")
                break

            # -------------------------------------------------
            # Step 7: Step size / line search
            # -------------------------------------------------
            if self.line_search is None:
                if use_backtracking:
                    step = alpha
                    c1 = float(kwargs.get("armijo_c", 1e-6))
                    tau = float(kwargs.get("backtracking_tau", 0.5))
                    min_step = float(kwargs.get("min_step", 1e-10))
                    max_backtracks = int(kwargs.get("max_backtracks", 30))

                    # For steepest descent, grad^T (-grad) = -||grad||^2
                    directional_derivative = -grad_norm ** 2

                    accepted = False
                    for _ in range(max_backtracks):
                        x_trial = x - step * grad
                        u_trial = self.state_solver(model=self.model, x=x_trial, **kwargs)
                        J_trial = float(self.model.objective(u_trial, x_trial))

                        if verbose:
                            print(f"  backtrack step={step:.3e}")
                            print(f"  trial objective={J_trial:.12e}")

                        if np.isfinite(J_trial) and J_trial <= J + c1 * step * directional_derivative:
                            accepted = True
                            break

                        step *= tau

                        if step < min_step:
                            break

                    if not accepted:
                        # Keep optimization alive, but skip the unstable update.
                        step = 0.0

                        if store_history:
                            history["step_size"].append(step)

                        if verbose:
                            print("Backtracking failed to find a stable step. Skipping update and continuing.")

                        k += 1
                        continue
                else:
                    # CHANGED: plain fixed-step gradient descent when backtracking is disabled
                    step = alpha
                    if verbose:
                        print(f"  fixed step={step:.3e}")

            else:
                step = self.line_search(
                    model=self.model,
                    x=x,
                    u=u,
                    J=J,
                    grad=grad,
                    state_solver=self.state_solver,
                    **kwargs,
                )

            if store_history:
                history["step_size"].append(step)

            # -------------------------------------------------
            # Step 8: Control update
            # -------------------------------------------------
            x = x - step * grad

            k += 1

        return OptimizationResult(
            x_star=x,
            objective_value=float(J),
            gradient_norm=float(np.linalg.norm(grad)),
            iterations=k,
            converged=converged,
            history=history,
        )