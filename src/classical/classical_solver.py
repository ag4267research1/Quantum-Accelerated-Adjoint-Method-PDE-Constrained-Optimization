import numpy as np


def state_solver(model, x, max_iter=25, tol=1e-8, **kwargs):
    """
    Solve the nonlinear PDE state equation

        c(u, x) = 0

    where

        u : state variable (temperature)
        x : control variable (forcing coefficients)

    The state equation is nonlinear because of the radiation term,
    so Newton iterations are used.

    Newton iteration:

        J(u_k) Δu = -r(u_k)

        u_{k+1} = u_k + α Δu

    where

        r(u) = PDE residual
        J(u) = Jacobian ∂c/∂u
        α    = damping parameter

    Parameters
    ----------
    model : HeatModel
        Object containing the PDE residual and Jacobian.

    x : ndarray
        Control variable.

    max_iter : int
        Maximum number of Newton iterations.

    tol : float
        Convergence tolerance based on residual norm.

    Returns
    -------
    u : ndarray
        State variable (temperature solution).
    """

    # Initial guess for the state
    u = model.initial_state()

    # Damping factor for Newton updates
    alpha = 0.5

    for k in range(max_iter):

        # -------------------------------------------------
        # Step 1: Compute PDE residual r(u,x)
        # -------------------------------------------------
        r = model.residual(u, x)

        res_norm = np.linalg.norm(r)

        if res_norm < tol:
            break

        # -------------------------------------------------
        # Step 2: Assemble Jacobian J = ∂c/∂u
        # -------------------------------------------------
        J = model.jacobian(u, x)

        # Small regularization to improve numerical stability
        J = J + 1e-8 * np.eye(len(u))

        # -------------------------------------------------
        # Step 3: Solve Newton step
        #
        #        J Δu = -r
        # -------------------------------------------------
        delta_u = np.linalg.solve(J, -r)

        # -------------------------------------------------
        # Step 4: Update state
        # -------------------------------------------------
        u = u + alpha * delta_u

    return u


def adjoint_solver(A, rhs, **kwargs):
    """
    Solve the adjoint equation

        A^T p = rhs

    where

        A   = ∂c/∂u      (Jacobian of PDE constraint)
        rhs = ∂J/∂u      (objective derivative with respect to state)

    Parameters
    ----------
    A : ndarray
        Jacobian matrix ∂c/∂u

    rhs : ndarray
        Gradient of the objective with respect to the state.

    Returns
    -------
    p : ndarray
        Adjoint vector.
    """

    p = np.linalg.solve(A.T, rhs)

    return p


def inner_product(left, right, **kwargs):
    """
    Compute vector inner product

        <left, right>

    This is used during gradient assembly in the adjoint method.

    Parameters
    ----------
    left : ndarray
        First vector.

    right : ndarray
        Second vector.

    Returns
    -------
    float
        Dot product of the two vectors.
    """

    return float(np.dot(left, right))