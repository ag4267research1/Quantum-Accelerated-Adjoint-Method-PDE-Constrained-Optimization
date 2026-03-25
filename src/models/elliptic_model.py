"""
1D Elliptic PDE Model (Two-variable formulation)

We use ONLY:
    - u : state variable
    - x : control variable

Continuous PDE:
    -(kappa(x) u'(x))' + c(x) u(x) = B x + f(x)

Boundary conditions:
    u(0) = 0, u(1) = 0

Discrete form:
    A u = b(x)

No extra symbols (no s, no y).
"""

import numpy as np


class EllipticModel:
    def __init__(self, config):

        self.config = config

        # --------------------------------------------------
        # Grid control coordinate)
        # --------------------------------------------------

        self.n = config.get("grid_size", 32)
        self.h = 1.0 / (self.n + 1)

        # x = domain grid
        self.x = np.linspace(self.h, 1.0 - self.h, self.n)

        self.num_dofs = self.n

        self.exp_type = config.get("experiment_type", "exp1")
        self.alpha = float(config.get("alpha", 1e-4))

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build_system(self, x):
        """
        Build linear system A u = b(x)
        """

        A = self._build_operator()
        b = self._build_rhs(x)

        return A, b

    def initial_state(self):
        return np.zeros(self.num_dofs)

    def residual(self, u, x):
        A, b = self.build_system(x)
        return A @ u - b

    def jacobian(self, u, x):
        A, _ = self.build_system(x)
        return A

    def objective(self, u, x):
        u_d = self.desired_state()
        return 0.5 * np.linalg.norm(u - u_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

    def dJ_du(self, u, x):
        u_d = self.desired_state()
        return u - u_d

    def dJ_dx(self, u, x):
        return self.alpha * x

    def dc_dx_i(self, u, x, i):
        e_i = np.zeros(self.num_dofs)
        e_i[i] = 1.0
        return -self._apply_control(e_i)

    def desired_state(self):
        return np.sin(np.pi * self.x)

    # =========================================================
    # OPERATOR A
    # =========================================================

    def _build_operator(self):
        """
        Build symmetric finite-difference operator
        """

        n = self.n
        h = self.h

        kappa = self._kappa(self.x)
        c = self._c(self.x)

        A = np.zeros((n, n))

        # interface diffusion (to preserve symmetry)
        k_half = 0.5 * (kappa[:-1] + kappa[1:])

        for i in range(n):

            # left coefficient
            if i > 0:
                k_left = k_half[i - 1]
            else:
                k_left = kappa[i]

            # right coefficient
            if i < n - 1:
                k_right = k_half[i]
            else:
                k_right = kappa[i]

            A[i, i] = (k_left + k_right) / h**2 + c[i]

            if i > 0:
                A[i, i - 1] = -k_left / h**2

            if i < n - 1:
                A[i, i + 1] = -k_right / h**2

        return A

    # =========================================================
    # RHS
    # =========================================================

    def _build_rhs(self, x):
        """
        b = B x + f(x)
        """

        Bx = self._apply_control(x)
        f = self._f(self.x)

        return Bx + f

    # =========================================================
    # CONTROL
    # =========================================================

    def _apply_control(self, x):
        """
        Apply control operator
        """

        if self.exp_type == "exp4":
            mask = self._control_mask(self.x)
            return mask * x

        return x

    # =========================================================
    # COEFFICIENTS
    # =========================================================

    def _kappa(self, x):

        if self.exp_type == "exp2":
            return 1 + 0.5 * np.sin(2 * np.pi * x)

        if self.exp_type == "exp3":
            k = np.ones_like(x)
            k[x >= 0.5] = 1e2
            return k

        return np.ones_like(x)

    def _c(self, x):

        if self.exp_type == "exp2":
            return np.ones_like(x)

        return np.zeros_like(x)

    # =========================================================
    # SOURCE
    # =========================================================

    def _f(self, x):

        if self.exp_type == "exp5":
            return (np.pi**2) * np.sin(np.pi * x)

        return np.zeros_like(x)

    # =========================================================
    # CONTROL MASK
    # =========================================================

    def _control_mask(self, x):

        mask = np.zeros_like(x)
        mask[(x >= 0.2) & (x <= 0.4)] = 1.0
        return mask