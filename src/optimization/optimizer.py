
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
        alpha: float = 1e-2,
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
        }

        converged = False
        J = np.nan
        grad = np.zeros_like(x)

        for k in range(max_iter):
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
            # Quantum: returns a state
            # -------------------------------------------------
            p = self.adjoint_solver(A=A, rhs=g_u, model=self.model, u=u, x=x, **kwargs)

            # -------------------------------------------------
            # Step 5: Assemble reduced gradient
            # grad_i = dJ/dx_i - <p, dc/dx_i>
            # -------------------------------------------------
            grad = np.zeros_like(x, dtype=float)

            for i in range(len(x)):
                w_i = self.model.dc_dx_i(u, x, i)
                z_i = self.inner_product(left=w_i, right=p, model=self.model, u=u, x=x, i=i, **kwargs)
                grad[i] = g_x[i] - z_i

            grad_norm = float(np.linalg.norm(grad))

            if store_history:
                history["objective"].append(J)
                history["gradient_norm"].append(grad_norm)

            if verbose:
                print(
                    f"iter={k:04d} | J={J:.6e} | ||grad||={grad_norm:.6e}"
                )

            # -------------------------------------------------
            # Step 6: Stopping test
            # -------------------------------------------------
            if grad_norm <= tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {k}.")
                break

            # -------------------------------------------------
            # Step 7: Step size / line search
            # -------------------------------------------------
            if self.line_search is None:
                step = alpha
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

        return OptimizationResult(
            x_star=x,
            objective_value=float(J),
            gradient_norm=float(np.linalg.norm(grad)),
            iterations=k + 1,
            converged=converged,
            history=history,
        )