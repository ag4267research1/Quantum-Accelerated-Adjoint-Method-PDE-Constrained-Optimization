import numpy as np


class HeatModel:
    """
    Nonlinear 1D steady heat transfer model.

    Governing equation
    ------------------
        d²T/dy² = ε(T) (T⁴ − T_inf⁴) + h (T − T_inf) + forcing(x)

    where

        T(y)      : temperature field
        ε(T)      : nonlinear radiation coefficient
        T_inf     : ambient temperature
        h         : convection coefficient
        forcing(x): source term parameterized by a low-dimensional
                    control vector x

    Boundary conditions
    -------------------
        T(0) = T01
        T(1) = T02

    State variable
    --------------
    u : temperature values at the interior grid points

    Control variable
    ----------------
    x : vector of coefficients that parameterize the source term

    Notes
    -----
    This class is designed for PDE-constrained optimization and provides

        residual(u, x)    : PDE residual c(u, x)
        jacobian(u, x)    : state Jacobian ∂c/∂u
        dc_dx_i(u, x, i)  : derivative of residual wrt control x_i
        objective(u, x)   : scalar objective function
        dJ_du(u, x)       : derivative of objective wrt state
        dJ_dx(u, x)       : derivative of objective wrt control
        initial_state()   : initial guess for nonlinear solve
    """

    def __init__(
        self,
        n=100,
        nx=5,
        x1=0.0,
        x2=1.0,
        T01=0.0,
        T02=0.0,
        k=1.0,
        T_inf=50.0,
        h=0.5,
        source_scale=0.1,
        objective_type="tracking",
        beta=1e-3,
        target=None,
    ):
        """
        Parameters
        ----------
        n : int
            Number of state variables / interior spatial points.

        nx : int
            Number of control variables.

        x1, x2 : float
            Spatial domain endpoints.

        T01, T02 : float
            Dirichlet boundary temperatures.

        k : float
            Diffusion coefficient.

        T_inf : float
            Ambient temperature.

        h : float
            Convection coefficient.

        source_scale : float
            Scaling factor applied to the control source term.

            A smaller value is safer when running optimization.
            Large values can produce unrealistically strong forcing.

        objective_type : str
            Choice of objective function.

            "tracking" :
                J(u,x) = 1/2 ||u - target||² + (beta/2) ||x||²

            "energy" :
                J(u,x) = ∫ u dy

            The "tracking" objective is better for optimization because
            it makes the problem well-posed. The "energy" objective is
            useful for sensitivity studies.

        beta : float
            Control regularization parameter used in the tracking objective.

        target : ndarray or None
            Desired temperature profile for the tracking objective.
            If None, a default target profile is constructed.
        """

        self.n = n
        self.nx = nx

        self.x1 = x1
        self.x2 = x2

        self.T01 = T01
        self.T02 = T02

        self.k = k
        self.T_inf = T_inf
        self.h = h

        self.source_scale = source_scale

        self.objective_type = objective_type
        self.beta = beta

        # Grid spacing. The state is stored at cell centers.
        self.dy = (x2 - x1) / n

        # Cell-centered grid:
        # y_i = x1 + (i + 1/2) dy
        self.grid = np.linspace(x1, x2 - self.dy, n) + 0.5 * self.dy

        # Default target temperature profile used by the tracking objective.
        # This produces a smooth plateau-like target close to the ambient
        # temperature in the interior while respecting lower boundary values.
        if target is None:
            self.target = self._default_target()
        else:
            self.target = np.asarray(target, dtype=float)

    # --------------------------------------------------
    # Default target profile
    # --------------------------------------------------

    def _default_target(self):
        """
        Construct a smooth default target profile for optimization.

        The target is chosen to be near the ambient temperature in the
        interior and lower near the boundaries. This avoids pushing the
        optimizer toward extreme source values.

        Returns
        -------
        ndarray of shape (n,)
        """

        y = (self.grid - self.x1) / (self.x2 - self.x1)

        # Smooth plateau profile between the two boundaries
        profile = 4.0 * y * (1.0 - y)

        return self.T_inf * profile

    # --------------------------------------------------
    # Radiation coefficient ε(T)
    # --------------------------------------------------

    def epsilon(self, T):
        """
        Nonlinear radiation coefficient

            ε(T) = (1 + 5 sin(3πT/200) + exp(0.02T)) × 10⁻⁴

        The exponential argument is clipped to avoid overflow.
        """

        return (
            1.0
            + 5.0 * np.sin((3.0 * np.pi / 200.0) * T)
            + np.exp(np.clip(0.02 * T, -50.0, 50.0))
        ) * 1e-4

    # --------------------------------------------------
    # Derivative of ε(T)
    # --------------------------------------------------

    def epsilon_prime(self, T):
        """
        Derivative of ε(T) with respect to temperature.
        """

        return (
            5.0 * (3.0 * np.pi / 200.0) * np.cos((3.0 * np.pi / 200.0) * T)
            + 0.02 * np.exp(np.clip(0.02 * T, -50.0, 50.0))
        ) * 1e-4

    # --------------------------------------------------
    # Source basis function
    # --------------------------------------------------

    def source_basis(self, i, j):
        """
        Polynomial basis used for the control source term.

        Each control coefficient x_j contributes

            x_j * φ_j(y_i)

        with

            φ_j(y_i) = source_scale * y_i^j

        Parameters
        ----------
        i : int
            Spatial index.

        j : int
            Control index.

        Returns
        -------
        float
        """

        y_i = self.grid[i]
        return self.source_scale * (y_i ** j)

    # --------------------------------------------------
    # Full source term at grid point i
    # --------------------------------------------------

    def source_term(self, i, x):
        """
        Evaluate the source term contribution at grid index i.

        Parameters
        ----------
        i : int
            Spatial index.

        x : ndarray of shape (nx,)
            Control vector.

        Returns
        -------
        float
        """

        value = 0.0

        for j in range(self.nx):
            value += x[j] * self.source_basis(i, j)

        return value

    # --------------------------------------------------
    # PDE residual c(u, x)
    # --------------------------------------------------

    def residual(self, u, x):
        """
        Compute the discretized PDE residual.

        Residual definition
        -------------------
            c(u,x) = diffusion + radiation + convection + forcing

        with

            diffusion = k (u_{i-1} - 2u_i + u_{i+1}) / dy²
            radiation = ε(u_i) (T_inf⁴ - u_i⁴)
            convection = h (T_inf - u_i)
            forcing    = source_term(i, x)

        Returns
        -------
        ndarray of shape (n,)
        """

        r = np.zeros_like(u)

        dy2 = self.dy ** 2

        for i in range(self.n):

            # Neighbor values with Dirichlet boundary conditions
            u_im1 = self.T01 if i == 0 else u[i - 1]
            u_ip1 = self.T02 if i == self.n - 1 else u[i + 1]

            # Clip temperature in nonlinear evaluations to avoid overflow
            u_i = np.clip(u[i], -500.0, 500.0)

            # Diffusion contribution
            diffusion = self.k * (u_im1 - 2.0 * u_i + u_ip1) / dy2

            # Radiation contribution
            radiation = self.epsilon(u_i) * (self.T_inf ** 4 - u_i ** 4)

            # Convection contribution
            convection = self.h * (self.T_inf - u_i)

            # Control forcing contribution
            forcing = self.source_term(i, x)

            r[i] = diffusion + radiation + convection + forcing

        return r

    # --------------------------------------------------
    # Jacobian ∂c/∂u
    # --------------------------------------------------

    def jacobian(self, u, x):
        """
        Jacobian of the residual with respect to the state variable u.

        For each row i, the stencil contributions are

            ∂c_i/∂u_{i-1} =  k / dy²
            ∂c_i/∂u_{i+1} =  k / dy²

        and the diagonal term is

            ∂c_i/∂u_i =
                -2k/dy²
                + ε'(u_i) (T_inf⁴ - u_i⁴)
                - 4 ε(u_i) u_i³
                - h

        Returns
        -------
        ndarray of shape (n, n)
        """

        A = np.zeros((self.n, self.n))

        dy2 = self.dy ** 2

        for i in range(self.n):

            u_i = np.clip(u[i], -500.0, 500.0)

            eps = self.epsilon(u_i)
            deps = self.epsilon_prime(u_i)

            if i > 0:
                A[i, i - 1] = self.k / dy2

            if i < self.n - 1:
                A[i, i + 1] = self.k / dy2

            A[i, i] = (
                -2.0 * self.k / dy2
                + deps * (self.T_inf ** 4 - u_i ** 4)
                - 4.0 * eps * u_i ** 3
                - self.h
            )

        return A

    # --------------------------------------------------
    # Derivative of residual with respect to x_i
    # --------------------------------------------------

    def dc_dx_i(self, u, x, i):
        """
        Compute the derivative of the residual with respect to control x_i.

        Since the source term is linear in x_i,

            ∂c/∂x_i = φ_i(y)

        Parameters
        ----------
        u : ndarray
            State variable. Included for interface consistency.

        x : ndarray
            Control variable. Included for interface consistency.

        i : int
            Control index.

        Returns
        -------
        ndarray of shape (n,)
        """

        vec = np.zeros(self.n)

        for row in range(self.n):
            vec[row] = self.source_basis(row, i)

        return vec

    # --------------------------------------------------
    # Objective function
    # --------------------------------------------------

    def objective(self, u, x):
        """
        Compute the scalar objective.

        Available objective choices
        ---------------------------
        tracking:
            J(u,x) = 1/2 ||u - target||² + (beta/2) ||x||²

        energy:
            J(u,x) = ∫ u dy

        Returns
        -------
        float
        """

        if self.objective_type == "tracking":
            misfit = 0.5 * np.linalg.norm(u - self.target) ** 2
            regularization = 0.5 * self.beta * np.linalg.norm(x) ** 2
            return misfit + regularization

        if self.objective_type == "energy":
            return np.sum(u) * self.dy

        raise ValueError(f"Unknown objective_type: {self.objective_type}")

    # --------------------------------------------------
    # Derivative of objective with respect to u
    # --------------------------------------------------

    def dJ_du(self, u, x):
        """
        Compute ∂J/∂u.

        Returns
        -------
        ndarray of shape (n,)
        """

        if self.objective_type == "tracking":
            return u - self.target

        if self.objective_type == "energy":
            return np.ones(self.n) * self.dy

        raise ValueError(f"Unknown objective_type: {self.objective_type}")

    # --------------------------------------------------
    # Derivative of objective with respect to x
    # --------------------------------------------------

    def dJ_dx(self, u, x):
        """
        Compute ∂J/∂x.

        Returns
        -------
        ndarray of shape (nx,)
        """

        if self.objective_type == "tracking":
            return self.beta * x

        if self.objective_type == "energy":
            return np.zeros(self.nx)

        raise ValueError(f"Unknown objective_type: {self.objective_type}")

    # --------------------------------------------------
    # Initial guess for nonlinear solve
    # --------------------------------------------------

    def initial_state(self):
        """
        Initial temperature field used by the nonlinear solver.

        A uniform ambient-temperature initial guess is used.
        """

        return np.ones(self.n) * self.T_inf