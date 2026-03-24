"""
1D Elliptic PDE Model

Governing equation
------------------
We consider the 1D elliptic PDE

    -(kappa(x) y'(x))' + c(x) y(x) = B u(x) + f(x),   x in (0,1),

with homogeneous Dirichlet boundary conditions

    y(0) = 0,
    y(1) = 0.

Here

    y(x) : state variable
    u(x) : control variable
    kappa(x) : diffusion coefficient
    c(x) : reaction coefficient
    B : control operator
    f(x) : source term

Optimization setting
--------------------
The model is designed for PDE-constrained optimization with objective

    J(y, u) = 1/2 ||y - y_d||^2 + (alpha/2) ||u||^2,

where

    y_d : desired state
    alpha : regularization parameter

This class provides the interface expected by the existing optimizer
and solvers, namely:

    build_system(u)      : construct A y = b
    initial_state()      : initial guess for the state solver
    residual(y, u)       : state equation residual
    jacobian(y, u)       : Jacobian wrt the state
    objective(y, u)      : scalar objective function
    dJ_du(y, u)          : derivative of objective wrt state
    dJ_dx(y, u)          : derivative of objective wrt control
    dc_dx_i(y, u, i)     : derivative of residual wrt control component i

Experiments
-----------
The model supports five experiment types:

    exp1 : baseline Poisson-type operator
    exp2 : diffusion-reaction with spatially varying kappa and nonzero c
    exp3 : heterogeneous diffusion coefficient with jump
    exp4 : localized control via a control mask
    exp5 : source perturbation via nonzero f

Important design choice
-----------------------
We do NOT modify the optimizer or the solvers.
Everything is encoded in the linear system

    A y = b

so that the rest of the pipeline can remain unchanged.
"""

import numpy as np


class EllipticModel:
    """
    1D elliptic PDE model compatible with the existing optimization framework.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
                - grid_size
                - experiment_type
                - alpha
                - other experiment-specific options
        """

        self.config = config

        # --------------------------------------------------
        # Grid definition
        # --------------------------------------------------
        #
        # We use n interior points on the interval (0,1).
        # The spacing is
        #
        #     h = 1 / (n + 1),
        #
        # because the two boundary points x=0 and x=1 are excluded from
        # the state vector and treated through Dirichlet boundary conditions.
        # --------------------------------------------------

        self.n = config.get("grid_size", 32)
        self.h = 1.0 / (self.n + 1)

        # Number of degrees of freedom in the state vector
        self.num_dofs = self.n

        # --------------------------------------------------
        # Experiment type
        # --------------------------------------------------
        #
        # This determines which coefficients/operators are active.
        # --------------------------------------------------

        self.exp_type = config.get("experiment_type", "exp1")

        # --------------------------------------------------
        # Objective regularization parameter
        # --------------------------------------------------

        self.alpha = float(config.get("alpha", 1e-4))

        # --------------------------------------------------
        # Interior grid points
        # --------------------------------------------------
        #
        # These are the coordinates at which the discrete state y is stored.
        # --------------------------------------------------

        self.x = np.linspace(self.h, 1.0 - self.h, self.n)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build_system(self, u):
        """
        Build the linear system

            A y = b

        corresponding to the discretized elliptic PDE.

        Parameters
        ----------
        u : ndarray of shape (n,)
            Control vector.

        Returns
        -------
        A : ndarray of shape (n, n)
            Discrete elliptic operator.

        b : ndarray of shape (n,)
            Right-hand side containing control and source contributions.
        """

        A = self._build_operator()
        b = self._build_rhs(u)

        return A, b

    def initial_state(self):
        """
        Return an initial guess for the state variable y.

        This is used by the classical state solver. Since the PDE is linear,
        a zero initial guess is sufficient.

        Returns
        -------
        ndarray of shape (n,)
        """

        return np.zeros(self.num_dofs)

    def residual(self, y, x):
        """
        Compute the residual of the discretized state equation.

        The PDE is represented as

            A y = b(x),

        so the residual is

            c(y, x) = A y - b(x).

        Parameters
        ----------
        y : ndarray
            State vector.

        x : ndarray
            Control vector.

        Returns
        -------
        ndarray of shape (n,)
        """

        A, b = self.build_system(x)
        return A @ y - b

    def jacobian(self, y, x):
        """
        Jacobian of the residual with respect to the state variable y.

        Since the PDE is linear in y, this Jacobian is simply the matrix A.

        Parameters
        ----------
        y : ndarray
            State vector (included for interface compatibility).

        x : ndarray
            Control vector (included for interface compatibility).

        Returns
        -------
        ndarray of shape (n, n)
        """

        A, _ = self.build_system(x)
        return A

    def objective(self, y, x):
        """
        Compute the optimization objective

            J(y, x) = 1/2 ||y - y_d||^2 + (alpha/2) ||x||^2.

        The first term is a tracking term that pushes the state toward
        a desired profile y_d. The second term penalizes large controls.

        Parameters
        ----------
        y : ndarray
            State vector.

        x : ndarray
            Control vector.

        Returns
        -------
        float
        """

        y_d = self.desired_state()
        return 0.5 * np.linalg.norm(y - y_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

    def dJ_du(self, y, x):
        """
        Derivative of the objective with respect to the state y.

        For

            J(y, x) = 1/2 ||y - y_d||^2 + ...

        we have

            dJ/du = y - y_d.

        Parameters
        ----------
        y : ndarray
            State vector.

        x : ndarray
            Control vector (unused here but kept for interface compatibility).

        Returns
        -------
        ndarray of shape (n,)
        """

        y_d = self.desired_state()
        return y - y_d

    def dJ_dx(self, y, x):
        """
        Derivative of the objective with respect to the control x.

        For the quadratic regularization term

            (alpha/2) ||x||^2,

        the derivative is

            dJ/dx = alpha * x.

        Parameters
        ----------
        y : ndarray
            State vector (unused here but kept for interface compatibility).

        x : ndarray
            Control vector.

        Returns
        -------
        ndarray of shape (n,)
        """

        return self.alpha * x

    def dc_dx_i(self, y, x, i):
        """
        Derivative of the state residual c(y, x) with respect to control x_i.

        The discrete state equation is

            c(y, x) = A y - (B x + f).

        Therefore,

            d c / d x_i = - B e_i,

        where e_i is the i-th canonical basis vector.

        Parameters
        ----------
        y : ndarray
            State vector (unused here but kept for interface compatibility).

        x : ndarray
            Control vector (unused here but kept for interface compatibility).

        i : int
            Control index.

        Returns
        -------
        ndarray of shape (n,)
        """

        e_i = np.zeros(self.num_dofs)
        e_i[i] = 1.0

        return -self._apply_control(e_i)

    def desired_state(self):
        """
        Desired state y_d used in the tracking objective.

        For now we use a smooth manufactured target

            y_d(x) = sin(pi x),

        which is a simple nontrivial profile for testing optimization.

        Returns
        -------
        ndarray of shape (n,)
        """

        return np.sin(np.pi * self.x)

    # =========================================================
    # BUILD OPERATOR A
    # =========================================================

    def _build_operator(self):
        """
        Build the discrete elliptic operator A.

        Continuous operator
        -------------------
        The PDE operator is

            -(kappa(x) y')' + c(x) y.

        Discretization
        --------------
        We use a symmetric flux-form finite difference discretization.

        For interior node i, the diffusion term is approximated by

            -(kappa y')'(x_i)
            ≈ [ -kappa_{i+1/2}(y_{i+1} - y_i)
                + kappa_{i-1/2}(y_i - y_{i-1}) ] / h^2.

        This leads to the tridiagonal stencil

            A[i,i]   = (k_left + k_right)/h^2 + c_i
            A[i,i-1] = -k_left / h^2
            A[i,i+1] = -k_right / h^2

        where k_left and k_right are interface values of kappa.

        Why flux form?
        --------------
        If kappa varies in space, using pointwise values directly can
        produce a nonsymmetric matrix. The flux-form stencil preserves
        symmetry/Hermiticity for real-valued kappa, which is important
        for HHL-based quantum linear solvers.

        Returns
        -------
        ndarray of shape (n, n)
        """

        n = self.n
        h = self.h

        kappa = self._kappa(self.x)
        c = self._c(self.x)

        A = np.zeros((n, n))

        # --------------------------------------------------
        # Compute interface diffusion coefficients
        # --------------------------------------------------
        #
        # kappa_half[i] approximates kappa at x_{i+1/2}.
        # Arithmetic averaging is used here for simplicity.
        # --------------------------------------------------

        kappa_half = np.zeros(n - 1)
        for i in range(n - 1):
            kappa_half[i] = 0.5 * (kappa[i] + kappa[i + 1])

        for i in range(n):

            # Left interface coefficient kappa_{i-1/2}
            if i > 0:
                k_left = kappa_half[i - 1]
            else:
                # Near the left boundary we use the node value
                k_left = kappa[i]

            # Right interface coefficient kappa_{i+1/2}
            if i < n - 1:
                k_right = kappa_half[i]
            else:
                # Near the right boundary we use the node value
                k_right = kappa[i]

            # Diagonal entry
            A[i, i] = (k_left + k_right) / h**2 + c[i]

            # Left off-diagonal
            if i > 0:
                A[i, i - 1] = -k_left / h**2

            # Right off-diagonal
            if i < n - 1:
                A[i, i + 1] = -k_right / h**2

        return A

    # =========================================================
    # BUILD RHS
    # =========================================================

    def _build_rhs(self, u):
        """
        Build the right-hand side

            b = B u + f.

        Here
        ----
        - B u represents the control contribution
        - f represents the source term contribution

        Parameters
        ----------
        u : ndarray of shape (n,)
            Control vector.

        Returns
        -------
        ndarray of shape (n,)
        """

        Bu = self._apply_control(u)
        f = self._f(self.x)

        return Bu + f

    # =========================================================
    # CONTROL OPERATOR
    # =========================================================

    def _apply_control(self, u):
        """
        Apply the control operator B to the control vector u.

        Default case
        ------------
        For most experiments, the control acts everywhere, so

            B u = u.

        Localized control
        -----------------
        In experiment exp4, the control is restricted to a subregion
        omega = [0.2, 0.4], implemented through a binary mask:

            B u = chi_omega * u.

        Parameters
        ----------
        u : ndarray of shape (n,)

        Returns
        -------
        ndarray of shape (n,)
        """

        if self.exp_type == "exp4":
            mask = self._control_mask(self.x)
            return mask * u

        return u

    # =========================================================
    # COEFFICIENTS
    # =========================================================

    def _kappa(self, x):
        """
        Diffusion coefficient kappa(x).

        Experiment-dependent definitions
        --------------------------------
        exp1, exp4, exp5:
            constant diffusion

                kappa(x) = 1

        exp2:
            smooth variable diffusion

                kappa(x) = 1 + 0.5 sin(2 pi x)

        exp3:
            heterogeneous medium with sharp jump

                kappa(x) = 1      for x < 0.5
                kappa(x) = 100    for x >= 0.5

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """

        if self.exp_type == "exp2":
            return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)

        if self.exp_type == "exp3":
            kappa = np.ones_like(x)
            kappa[x >= 0.5] = 1e2
            return kappa

        return np.ones_like(x)

    def _c(self, x):
        """
        Reaction coefficient c(x).

        Experiment-dependent definitions
        --------------------------------
        exp2:
            reaction term is active

                c(x) = 1

        all other experiments:
            no reaction term

                c(x) = 0

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """

        if self.exp_type == "exp2":
            return np.ones_like(x)

        return np.zeros_like(x)

    # =========================================================
    # SOURCE TERM
    # =========================================================

    def _f(self, x):
        """
        Source term f(x).

        Experiment-dependent definitions
        --------------------------------
        exp5:
            nonzero manufactured source

                f(x) = pi^2 sin(pi x)

        all other experiments:
            zero source

                f(x) = 0

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """

        if self.exp_type == "exp5":
            return (np.pi ** 2) * np.sin(np.pi * x)

        return np.zeros_like(x)

    # =========================================================
    # CONTROL MASK
    # =========================================================

    def _control_mask(self, x):
        """
        Construct the localized control mask chi_omega.

        The active control region is

            omega = [0.2, 0.4].

        Thus

            chi_omega(x) = 1,  if x in omega
            chi_omega(x) = 0,  otherwise

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """

        mask = np.zeros_like(x)
        mask[(x >= 0.2) & (x <= 0.4)] = 1.0

        return mask