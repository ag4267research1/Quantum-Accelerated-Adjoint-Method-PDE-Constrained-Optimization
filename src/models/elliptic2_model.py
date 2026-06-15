import numpy as np


class Elliptic2Model:
    """
    One-dimensional elliptic PDE-constrained optimization problem.

    Strong form:
        -(kappa(x) u'(x))' + mu * u(x) = B x + f(x),   x in (0,1)
        u(0) = u(1) = 0

    After second-order FD discretization on a uniform grid with n
    interior points and mesh size h = 1/(n+1), the raw discrete system is

        A_raw u = b_raw(x)

    where A_raw has diagonal entries O(kappa/h^2).

    To control the condition number and ensure HHL succeeds at all grid
    sizes, a shift mu * I is added to A_raw before normalization, where
    mu = lambda_max(A_raw) is the largest eigenvalue. This gives:

        cond(A_raw + mu*I) <= 2   for all n and all five experiments.

    The shifted system is then normalized by its spectral norm:

        A    = (A_raw + mu*I) / ||A_raw + mu*I||_2    so ||A||_2 = 1
        b(x) = b_raw(x) / ||A_raw + mu*I||_2

    The solution u of the shifted system differs from the original but
    the optimization problem is posed on the shifted system throughout,
    so all gradients and adjoint quantities remain consistent.

    Public API:
        build_system(x)   -> (A, b)
        jacobian(u, x)    -> A
        residual(u, x)    -> A u - b
        objective(u, x)   -> scalar
        dJ_du(u, x)       -> n-vector
        dJ_dx(u, x)       -> nx-vector
        dc_dx_i(u, x, i)  -> n-vector
        desired_state()   -> n-vector
        initial_state()   -> n-vector of zeros
        num_dofs          -> nx
        exp_type          -> string
        x                 -> grid points
    """

    def __init__(self, config):

        self.config   = config
        self.n        = int(config.get("grid_size", 4))
        self.h        = 1.0 / (self.n + 1)
        self.x        = np.linspace(self.h, 1.0 - self.h, self.n)

        model_cfg     = config.get("model", {})
        self.nx       = int(model_cfg.get("nx", 4))
        self.num_dofs = self.nx

        self.exp_type = config.get("experiment_type", "exp1")
        self.alpha    = float(config.get("alpha", 1e-2))

        self._desired_state_cache = None

        # Compute shift mu = lambda_max(A_raw) and spectral norm once.
        A_raw        = self._build_A_raw()
        lam_max      = float(np.linalg.eigvalsh(A_raw)[-1])
        self._mu     = lam_max
        A_sh         = A_raw + self._mu * np.eye(self.n)
        self._A_norm = float(np.linalg.norm(A_sh, ord=2))

        # Control operator B (n x nx)
        self.B = self._build_B()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build_system(self, x):
        """Return the SCALED system  A u = b(x)  with ||A||_2 = 1."""
        A_raw  = self._build_A_raw()
        A_sh   = A_raw + self._mu * np.eye(self.n)
        b_raw  = self._build_b_raw(x)
        return A_sh / self._A_norm, b_raw / self._A_norm

    def initial_state(self):
        return np.zeros(self.n)

    def residual(self, u, x):
        A, b = self.build_system(x)
        return A @ u - b

    def jacobian(self, u, x):
        A, _ = self.build_system(x)
        return A

    def objective(self, u, x):
        u_d = self.desired_state()
        return (0.5 * np.linalg.norm(u - u_d) ** 2
                + 0.5 * self.alpha * np.linalg.norm(x) ** 2)

    def dJ_du(self, u, x):
        return u - self.desired_state()

    def dJ_dx(self, u, x):
        return self.alpha * x

    def dc_dx_i(self, u, x, i):
        """
        Derivative of the SCALED residual c(u,x) = A u - b(x)
        with respect to x_i.

        b(x) = b_raw(x) / A_norm  =>  d b / d x_i = B[:,i] / A_norm
        So  dc/dx_i = -B[:,i] / A_norm.

        Note: the shift mu*I does not depend on x, so it does not
        contribute to dc/dx_i.
        """
        return -self.B[:, i] / self._A_norm

    def desired_state(self):
        """u_d = 2 * u_ref,  u_ref = A^{-1} b(x_ref),  x_ref = ones(nx)."""
        if self._desired_state_cache is not None:
            return self._desired_state_cache
        x_ref = np.ones(self.nx)
        A, b  = self.build_system(x_ref)
        u_ref = np.linalg.solve(A, b)
        self._desired_state_cache = 2.0 * u_ref
        return self._desired_state_cache

    # =========================================================
    # INTERNAL: RAW (UNSCALED) OPERATOR
    # =========================================================

    def _build_A_raw(self):
        """
        Assemble A_raw for -(kappa u')' using standard second-order
        finite differences with symmetric interface averaging.
        """
        n, h   = self.n, self.h
        kappa  = self._kappa(self.x)
        A      = np.zeros((n, n))
        k_half = 0.5 * (kappa[:-1] + kappa[1:])
        for i in range(n):
            k_left  = k_half[i - 1] if i > 0     else kappa[i]
            k_right = k_half[i]     if i < n - 1 else kappa[i]
            A[i, i] = (k_left + k_right) / h ** 2
            if i > 0:     A[i, i - 1] = -k_left  / h ** 2
            if i < n - 1: A[i, i + 1] = -k_right / h ** 2
        return A

    # =========================================================
    # INTERNAL: RAW RHS
    # =========================================================

    def _build_b_raw(self, x):
        return self.B @ x + self._f(self.x)

    # =========================================================
    # INTERNAL: PDE COEFFICIENTS
    # =========================================================

    def _kappa(self, x):
        if self.exp_type == "exp2":
            return 1.0 + 0.3 * np.sin(2.0 * np.pi * x) ** 2
        if self.exp_type == "exp3":
            return 1.0 + 0.5 * np.tanh(10.0 * (x - 0.5))
        return np.ones_like(x)

    def _f(self, x):
        if self.exp_type == "exp5":
            return np.sin(np.pi * x)
        return np.zeros_like(x)

    # =========================================================
    # INTERNAL: CONTROL OPERATOR  B  (n x nx)
    # =========================================================

    def _build_B(self):
        B = np.zeros((self.n, self.nx))
        for j in range(self.nx):
            lo = j       / self.nx
            hi = (j + 1) / self.nx
            in_block = (self.x >= lo) & (self.x < hi)
            cnt = int(in_block.sum())
            if cnt > 0:
                B[in_block, j] = 1.0 / np.sqrt(cnt)
        if self.exp_type == "exp4":
            active = (self.x >= 0.25) & (self.x <= 0.75)
            B[~active, :] = 0.0
        return B