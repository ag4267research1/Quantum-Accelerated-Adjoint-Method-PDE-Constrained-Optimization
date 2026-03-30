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
            "condition_number": [],   # CHANGED: store cond(A) at each iteration
        }

        converged = False
        J = np.nan
        grad = np.zeros_like(x)

        # CHANGED: initialize grad_norm so we can use a while-loop stopping rule.
        grad_norm = np.inf

        # CHANGED: use a while loop so the optimization continues only while
        # both conditions hold:
        #   (1) k < max_iter
        #   (2) grad_norm >= 1e-3
        # This means the method stops when either max_iter is reached
        # or the gradient norm drops below 1e-3.
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

            # CHANGED: compute condition number of the Jacobian for plotting
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
            # Quantum: returns a quantum state and scaling
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
            # Quantum solver may return (p_state, scale)
            if isinstance(adjoint_output, tuple):
                p_state, p_scale = adjoint_output
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
                    **kwargs
                )

                # reduced gradient component
                grad[i] = g_x[i] - p_scale * z_i

            grad_norm = float(np.linalg.norm(grad))

            # CHANGED: stop immediately if the gradient becomes non-finite.
            # This prevents a corrupted quantum step from propagating and
            # blowing up the control/state on the next update.
            if not np.isfinite(grad_norm) or not np.all(np.isfinite(grad)):
                if verbose:
                    print("Stopping: non-finite gradient detected.")
                break

            if store_history:
                history["objective"].append(J)
                history["gradient_norm"].append(grad_norm)
                history["condition_number"].append(cond_A)   # CHANGED

            if verbose:
                print(
                    f"iter={k:04d} | J={J:.6e} | ||grad||={grad_norm:.12e}"
                )

            # -------------------------------------------------
            # Step 6: Stopping test
            # -------------------------------------------------
            if grad_norm <= tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {k}.")
                break

            # CHANGED: also mark convergence if the while-loop threshold
            # grad_norm < 1e-3 is reached.
            if grad_norm < 1e-3:
                converged = True
                if verbose:
                    print(f"Converged at iteration {k} (gradient norm below 1e-3).")
                break

            # -------------------------------------------------
            # Step 7: Step size / line search
            # -------------------------------------------------
            if self.line_search is None:
                # CHANGED: replace the always-accept fixed step with a
                # safeguarded backtracking step. This keeps the original
                # "alpha as default step size" behavior, but now rejects
                # steps that increase the objective too much.
                step = alpha
                c1 = kwargs.get("armijo_c", 1e-4)
                tau = kwargs.get("backtracking_tau", 0.5)
                min_step = kwargs.get("min_step", 1e-8)
                max_backtracks = kwargs.get("max_backtracks", 20)

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
                    # CHANGED: reject the unstable update instead of taking
                    # a bad full step that can cause the optimization to blow up.
                    if store_history:
                        history["step_size"].append(0.0)
                    if verbose:
                        print("Stopping: backtracking failed to find a stable step.")
                    break
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

            # CHANGED: manual iteration counter for the while loop.
            k += 1

        return OptimizationResult(
            x_star=x,
            objective_value=float(J),
            gradient_norm=float(np.linalg.norm(grad)),
            iterations=k,
            converged=converged,
            history=history,
        )