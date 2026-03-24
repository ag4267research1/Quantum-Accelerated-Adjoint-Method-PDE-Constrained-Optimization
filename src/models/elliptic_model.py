# """
# 2D Elliptic PDE Model

# Implements:

#     -div(kappa(x) grad y) + c(x) y = B u + f   in Ω = (0,1)^2
#     y = 0 on ∂Ω

# We discretize using finite differences on a structured grid.

# Everything is reduced to:

#     A y = b

# IMPORTANT:
# We DO NOT modify optimizer or solvers.
# """

# import numpy as np


# class EllipticModel:
#     def __init__(self, config):

#         self.config = config

#         # grid size (n x n)
#         self.n = config.get("grid_size", 32)
#         self.h = 1.0 / (self.n + 1)

#         self.num_dofs = self.n * self.n

#         self.exp_type = config.get("experiment_type", "exp1")
#         self.alpha = float(config.get("alpha", 1e-4))

#         # generate grid
#         x = np.linspace(0, 1, self.n)
#         y = np.linspace(0, 1, self.n)
#         self.X, self.Y = np.meshgrid(x, y, indexing="ij")

#     # =========================================================
#     # PUBLIC API
#     # =========================================================

#     def build_system(self, u):
#         """
#         Build A y = b
#         """

#         A = self._build_operator()
#         b = self._build_rhs(u)

#         return A, b

#     def initial_state(self):
#         """
#         Return an initial guess for the state variable y.

#         The classical state solver expects the model to provide
#         an initial state iterate.
#         """

#         return np.zeros(self.num_dofs)

#     def residual(self, y, x):
#         """
#         Residual of the discretized state equation:

#             A y - b = 0

#         Here x is the control variable used to build b.
#         """

#         A, b = self.build_system(x)
#         return A @ y - b

#     def jacobian(self, y, x):
#         """
#         Jacobian of the state residual with respect to the state y.

#         Since the PDE is linear in y, this is simply A.
#         """

#         A, _ = self.build_system(x)
#         return A

#     def objective(self, y, x):
#         """
#         Objective functional:

#             J(y, x) = 1/2 ||y - y_d||^2 + alpha/2 ||x||^2
#         """

#         y_d = self.desired_state()
#         return 0.5 * np.linalg.norm(y - y_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

#     def dJ_du(self, y, x):
#         """
#         Partial derivative of J with respect to the state y.
#         """

#         y_d = self.desired_state()
#         return y - y_d

#     def dJ_dx(self, y, x):
#         """
#         Partial derivative of J with respect to the control x.
#         """

#         return self.alpha * x

#     def dc_dx_i(self, y, x, i):
#         """
#         Derivative of the state residual c(y, x) with respect to control x_i.

#         The state equation is:

#             A y - (B x + f) = 0

#         so

#             d c / d x_i = - B e_i

#         where e_i is the i-th canonical basis vector.
#         """

#         e_i = np.zeros(self.num_dofs)
#         e_i[i] = 1.0

#         return -self._apply_control(e_i)

#     def desired_state(self):
#         """
#         Desired state y_d used in the tracking objective.

#         For now we use the zero state as a simple default target.
#         This can easily be changed later if a nonzero target state
#         is needed for experiments.
#         """

#         return np.zeros(self.num_dofs)

#     # =========================================================
#     # BUILD OPERATOR A
#     # =========================================================

#     def _build_operator(self):
#         """
#         2D finite difference discretization of:

#             -div(kappa grad y) + c y
#         """

#         n = self.n
#         h = self.h
#         N = n * n

#         kappa = self._kappa(self.X, self.Y)
#         c = self._c(self.X, self.Y)

#         A = np.zeros((N, N))

#         def idx(i, j):
#             return i * n + j

#         for i in range(n):
#             for j in range(n):

#                 row = idx(i, j)

#                 # center coefficient
#                 A[row, row] = 4 * kappa[i, j] / h**2 + c[i, j]

#                 # neighbors (5-point stencil)

#                 # left
#                 if j > 0:
#                     A[row, idx(i, j - 1)] = -kappa[i, j] / h**2

#                 # right
#                 if j < n - 1:
#                     A[row, idx(i, j + 1)] = -kappa[i, j] / h**2

#                 # down
#                 if i > 0:
#                     A[row, idx(i - 1, j)] = -kappa[i, j] / h**2

#                 # up
#                 if i < n - 1:
#                     A[row, idx(i + 1, j)] = -kappa[i, j] / h**2

#         return A

#     # =========================================================
#     # BUILD RHS
#     # =========================================================

#     def _build_rhs(self, u):
#         """
#         b = B u + f
#         """

#         Bu = self._apply_control(u)

#         f = self._f(self.X, self.Y).reshape(-1)

#         return Bu + f

#     # =========================================================
#     # CONTROL OPERATOR
#     # =========================================================

#     def _apply_control(self, u):
#         """
#         Apply B u

#         exp4 → localized control
#         """

#         if self.exp_type == "exp4":
#             mask = self._control_mask(self.X, self.Y).reshape(-1)
#             return mask * u

#         return u

#     # =========================================================
#     # COEFFICIENTS
#     # =========================================================

#     def _kappa(self, X, Y):
#         """
#         Diffusion coefficient κ(x)
#         """

#         if self.exp_type == "exp2":
#             return 1 + 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

#         if self.exp_type == "exp3":
#             kappa = np.ones_like(X)
#             kappa[X >= 0.5] = 1e2
#             return kappa

#         return np.ones_like(X)

#     def _c(self, X, Y):
#         """
#         Reaction term
#         """

#         if self.exp_type == "exp2":
#             return np.ones_like(X)

#         return np.zeros_like(X)

#     # =========================================================
#     # SOURCE TERM
#     # =========================================================

#     def _f(self, X, Y):
#         """
#         Source term (exp5)
#         """

#         if self.exp_type == "exp5":
#             return 2 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

#         return np.zeros_like(X)

#     # =========================================================
#     # CONTROL MASK
#     # =========================================================

#     def _control_mask(self, X, Y):
#         """
#         Localized control ω = [0.2,0.4] x [0.2,0.4]
#         """

#         mask = np.zeros_like(X)

#         mask[
#             (X >= 0.2) & (X <= 0.4) &
#             (Y >= 0.2) & (Y <= 0.4)
#         ] = 1.0

#         return mask


# """
# 1D Elliptic PDE Model

# Implements:

#     -(kappa(x) y')' + c(x) y = B u + f   in Ω = (0,1)
#     y = 0 on ∂Ω

# We discretize using finite differences on a structured 1D grid.

# Everything is reduced to:

#     A y = b

# IMPORTANT:
# We DO NOT modify optimizer or solvers.
# """

# import numpy as np


# class EllipticModel:
#     def __init__(self, config):

#         self.config = config

#         # grid size (number of interior points)
#         self.n = config.get("grid_size", 32)
#         self.h = 1.0 / (self.n + 1)

#         self.num_dofs = self.n

#         self.exp_type = config.get("experiment_type", "exp1")
#         self.alpha = float(config.get("alpha", 1e-4))

#         # generate 1D grid of interior points
#         self.x = np.linspace(self.h, 1.0 - self.h, self.n)

#     # =========================================================
#     # PUBLIC API
#     # =========================================================

#     def build_system(self, u):
#         """
#         Build A y = b
#         """

#         A = self._build_operator()
#         b = self._build_rhs(u)

#         return A, b

#     def initial_state(self):
#         """
#         Return an initial guess for the state variable y.

#         The classical state solver expects the model to provide
#         an initial state iterate.
#         """

#         return np.zeros(self.num_dofs)

#     def residual(self, y, x):
#         """
#         Residual of the discretized state equation:

#             A y - b = 0

#         Here x is the control variable used to build b.
#         """

#         A, b = self.build_system(x)
#         return A @ y - b

#     def jacobian(self, y, x):
#         """
#         Jacobian of the state residual with respect to the state y.

#         Since the PDE is linear in y, this is simply A.
#         """

#         A, _ = self.build_system(x)
#         return A

#     def objective(self, y, x):
#         """
#         Objective functional:

#             J(y, x) = 1/2 ||y - y_d||^2 + alpha/2 ||x||^2
#         """

#         y_d = self.desired_state()
#         return 0.5 * np.linalg.norm(y - y_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

#     def dJ_du(self, y, x):
#         """
#         Partial derivative of J with respect to the state y.
#         """

#         y_d = self.desired_state()
#         return y - y_d

#     def dJ_dx(self, y, x):
#         """
#         Partial derivative of J with respect to the control x.
#         """

#         return self.alpha * x

#     def dc_dx_i(self, y, x, i):
#         """
#         Derivative of the state residual c(y, x) with respect to control x_i.

#         The state equation is:

#             A y - (B x + f) = 0

#         so

#             d c / d x_i = - B e_i

#         where e_i is the i-th canonical basis vector.
#         """

#         e_i = np.zeros(self.num_dofs)
#         e_i[i] = 1.0

#         return -self._apply_control(e_i)

#     def desired_state(self):
#         """
#         Desired state y_d used in the tracking objective.

#         For the baseline experiment we use a manufactured target.
#         You can change this later if needed.
#         """

#         return np.sin(np.pi * self.x)

#     # =========================================================
#     # BUILD OPERATOR A
#     # =========================================================

#     # def _build_operator(self):
#     #     """
#     #     1D finite difference discretization of:

#     #         -(kappa(x) y')' + c(x) y

#     #     For simplicity, this uses a standard tridiagonal stencil.
#     #     """

#     #     n = self.n
#     #     h = self.h

#     #     kappa = self._kappa(self.x)
#     #     c = self._c(self.x)

#     #     A = np.zeros((n, n))

#     #     for i in range(n):

#     #         # center coefficient
#     #         A[i, i] = 2.0 * kappa[i] / h**2 + c[i]

#     #         # left neighbor
#     #         if i > 0:
#     #             A[i, i - 1] = -kappa[i] / h**2

#     #         # right neighbor
#     #         if i < n - 1:
#     #             A[i, i + 1] = -kappa[i] / h**2

#     #     return A
    
#     def _build_operator(self):
#         """
#         1D finite difference discretization of:

#             -(kappa(x) y')' + c(x) y

#         using a symmetric flux-form stencil so that the
#         resulting matrix is Hermitian / symmetric.
#         """

#         n = self.n
#         h = self.h

#         kappa = self._kappa(self.x)
#         c = self._c(self.x)

#         A = np.zeros((n, n))

#         # interface values kappa_{i+1/2}
#         # simple arithmetic averaging
#         kappa_half = np.zeros(n - 1)
#         for i in range(n - 1):
#             kappa_half[i] = 0.5 * (kappa[i] + kappa[i + 1])

#         for i in range(n):

#             # left interface kappa_{i-1/2}
#             if i > 0:
#                 k_left = kappa_half[i - 1]
#             else:
#                 # boundary interface near x=0
#                 k_left = kappa[i]

#             # right interface kappa_{i+1/2}
#             if i < n - 1:
#                 k_right = kappa_half[i]
#             else:
#                 # boundary interface near x=1
#                 k_right = kappa[i]

#             # diagonal
#             A[i, i] = (k_left + k_right) / h**2 + c[i]

#             # left neighbor
#             if i > 0:
#                 A[i, i - 1] = -k_left / h**2

#             # right neighbor
#             if i < n - 1:
#                 A[i, i + 1] = -k_right / h**2

#         return A

#     # =========================================================
#     # BUILD RHS
#     # =========================================================

#     def _build_rhs(self, u):
#         """
#         b = B u + f
#         """

#         Bu = self._apply_control(u)
#         f = self._f(self.x)

#         return Bu + f

#     # =========================================================
#     # CONTROL OPERATOR
#     # =========================================================

#     def _apply_control(self, u):
#         """
#         Apply B u

#         exp4 → localized control
#         """

#         if self.exp_type == "exp4":
#             mask = self._control_mask(self.x)
#             return mask * u

#         return u

#     # =========================================================
#     # COEFFICIENTS
#     # =========================================================

#     def _kappa(self, x):
#         """
#         Diffusion coefficient κ(x)
#         """

#         if self.exp_type == "exp2":
#             return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)

#         if self.exp_type == "exp3":
#             kappa = np.ones_like(x)
#             kappa[x >= 0.5] = 1e2
#             return kappa

#         return np.ones_like(x)

#     def _c(self, x):
#         """
#         Reaction term
#         """

#         if self.exp_type == "exp2":
#             return np.ones_like(x)

#         return np.zeros_like(x)

#     # =========================================================
#     # SOURCE TERM
#     # =========================================================

#     def _f(self, x):
#         """
#         Source term (exp5)
#         """

#         if self.exp_type == "exp5":
#             return (np.pi ** 2) * np.sin(np.pi * x)

#         return np.zeros_like(x)

#     # =========================================================
#     # CONTROL MASK
#     # =========================================================

#     def _control_mask(self, x):
#         """
#         Localized control ω = [0.2,0.4]
#         """

#         mask = np.zeros_like(x)
#         mask[(x >= 0.2) & (x <= 0.4)] = 1.0

#         return mask


# """
# 2D Elliptic PDE Model

# Implements:

#     -div(kappa(x) grad y) + c(x) y = B u + f   in Ω = (0,1)^2
#     y = 0 on ∂Ω

# We discretize using finite differences on a structured grid.

# Everything is reduced to:

#     A y = b

# IMPORTANT:
# We DO NOT modify optimizer or solvers.
# """

# import numpy as np


# class EllipticModel:
#     def __init__(self, config):

#         self.config = config

#         # grid size (n x n)
#         self.n = config.get("grid_size", 32)
#         self.h = 1.0 / (self.n + 1)

#         self.num_dofs = self.n * self.n

#         self.exp_type = config.get("experiment_type", "exp1")
#         self.alpha = float(config.get("alpha", 1e-4))

#         # generate grid
#         x = np.linspace(0, 1, self.n)
#         y = np.linspace(0, 1, self.n)
#         self.X, self.Y = np.meshgrid(x, y, indexing="ij")

#     # =========================================================
#     # PUBLIC API
#     # =========================================================

#     def build_system(self, u):
#         """
#         Build A y = b
#         """

#         A = self._build_operator()
#         b = self._build_rhs(u)

#         return A, b

#     def initial_state(self):
#         """
#         Return an initial guess for the state variable y.

#         The classical state solver expects the model to provide
#         an initial state iterate.
#         """

#         return np.zeros(self.num_dofs)

#     def residual(self, y, x):
#         """
#         Residual of the discretized state equation:

#             A y - b = 0

#         Here x is the control variable used to build b.
#         """

#         A, b = self.build_system(x)
#         return A @ y - b

#     def jacobian(self, y, x):
#         """
#         Jacobian of the state residual with respect to the state y.

#         Since the PDE is linear in y, this is simply A.
#         """

#         A, _ = self.build_system(x)
#         return A

#     def objective(self, y, x):
#         """
#         Objective functional:

#             J(y, x) = 1/2 ||y - y_d||^2 + alpha/2 ||x||^2
#         """

#         y_d = self.desired_state()
#         return 0.5 * np.linalg.norm(y - y_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

#     def dJ_du(self, y, x):
#         """
#         Partial derivative of J with respect to the state y.
#         """

#         y_d = self.desired_state()
#         return y - y_d

#     def dJ_dx(self, y, x):
#         """
#         Partial derivative of J with respect to the control x.
#         """

#         return self.alpha * x

#     def dc_dx_i(self, y, x, i):
#         """
#         Derivative of the state residual c(y, x) with respect to control x_i.

#         The state equation is:

#             A y - (B x + f) = 0

#         so

#             d c / d x_i = - B e_i

#         where e_i is the i-th canonical basis vector.
#         """

#         e_i = np.zeros(self.num_dofs)
#         e_i[i] = 1.0

#         return -self._apply_control(e_i)

#     def desired_state(self):
#         """
#         Desired state y_d used in the tracking objective.

#         For now we use the zero state as a simple default target.
#         This can easily be changed later if a nonzero target state
#         is needed for experiments.
#         """

#         return np.zeros(self.num_dofs)

#     # =========================================================
#     # BUILD OPERATOR A
#     # =========================================================

#     def _build_operator(self):
#         """
#         2D finite difference discretization of:

#             -div(kappa grad y) + c y
#         """

#         n = self.n
#         h = self.h
#         N = n * n

#         kappa = self._kappa(self.X, self.Y)
#         c = self._c(self.X, self.Y)

#         A = np.zeros((N, N))

#         def idx(i, j):
#             return i * n + j

#         for i in range(n):
#             for j in range(n):

#                 row = idx(i, j)

#                 # center coefficient
#                 A[row, row] = 4 * kappa[i, j] / h**2 + c[i, j]

#                 # neighbors (5-point stencil)

#                 # left
#                 if j > 0:
#                     A[row, idx(i, j - 1)] = -kappa[i, j] / h**2

#                 # right
#                 if j < n - 1:
#                     A[row, idx(i, j + 1)] = -kappa[i, j] / h**2

#                 # down
#                 if i > 0:
#                     A[row, idx(i - 1, j)] = -kappa[i, j] / h**2

#                 # up
#                 if i < n - 1:
#                     A[row, idx(i + 1, j)] = -kappa[i, j] / h**2

#         return A

#     # =========================================================
#     # BUILD RHS
#     # =========================================================

#     def _build_rhs(self, u):
#         """
#         b = B u + f
#         """

#         Bu = self._apply_control(u)

#         f = self._f(self.X, self.Y).reshape(-1)

#         return Bu + f

#     # =========================================================
#     # CONTROL OPERATOR
#     # =========================================================

#     def _apply_control(self, u):
#         """
#         Apply B u

#         exp4 → localized control
#         """

#         if self.exp_type == "exp4":
#             mask = self._control_mask(self.X, self.Y).reshape(-1)
#             return mask * u

#         return u

#     # =========================================================
#     # COEFFICIENTS
#     # =========================================================

#     def _kappa(self, X, Y):
#         """
#         Diffusion coefficient κ(x)
#         """

#         if self.exp_type == "exp2":
#             return 1 + 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

#         if self.exp_type == "exp3":
#             kappa = np.ones_like(X)
#             kappa[X >= 0.5] = 1e2
#             return kappa

#         return np.ones_like(X)

#     def _c(self, X, Y):
#         """
#         Reaction term
#         """

#         if self.exp_type == "exp2":
#             return np.ones_like(X)

#         return np.zeros_like(X)

#     # =========================================================
#     # SOURCE TERM
#     # =========================================================

#     def _f(self, X, Y):
#         """
#         Source term (exp5)
#         """

#         if self.exp_type == "exp5":
#             return 2 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

#         return np.zeros_like(X)

#     # =========================================================
#     # CONTROL MASK
#     # =========================================================

#     def _control_mask(self, X, Y):
#         """
#         Localized control ω = [0.2,0.4] x [0.2,0.4]
#         """

#         mask = np.zeros_like(X)

#         mask[
#             (X >= 0.2) & (X <= 0.4) &
#             (Y >= 0.2) & (Y <= 0.4)
#         ] = 1.0

#         return mask


"""
1D Elliptic PDE Model

Implements:

    -(kappa(x) y')' + c(x) y = B u + f   in Ω = (0,1)
    y = 0 on ∂Ω

We discretize using finite differences on a structured 1D grid.

Everything is reduced to:

    A y = b

IMPORTANT:
We DO NOT modify optimizer or solvers.
"""

import numpy as np


class EllipticModel:
    def __init__(self, config):

        self.config = config

        # grid size (number of interior points)
        self.n = config.get("grid_size", 32)
        self.h = 1.0 / (self.n + 1)

        self.num_dofs = self.n

        self.exp_type = config.get("experiment_type", "exp1")
        self.alpha = float(config.get("alpha", 1e-4))

        # generate 1D grid of interior points
        self.x = np.linspace(self.h, 1.0 - self.h, self.n)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build_system(self, u):
        """
        Build A y = b
        """

        A = self._build_operator()
        b = self._build_rhs(u)

        return A, b

    def initial_state(self):
        """
        Return an initial guess for the state variable y.

        The classical state solver expects the model to provide
        an initial state iterate.
        """

        return np.zeros(self.num_dofs)

    def residual(self, y, x):
        """
        Residual of the discretized state equation:

            A y - b = 0

        Here x is the control variable used to build b.
        """

        A, b = self.build_system(x)
        return A @ y - b

    def jacobian(self, y, x):
        """
        Jacobian of the state residual with respect to the state y.

        Since the PDE is linear in y, this is simply A.
        """

        A, _ = self.build_system(x)
        return A

    def objective(self, y, x):
        """
        Objective functional:

            J(y, x) = 1/2 ||y - y_d||^2 + alpha/2 ||x||^2
        """

        y_d = self.desired_state()
        return 0.5 * np.linalg.norm(y - y_d) ** 2 + 0.5 * self.alpha * np.linalg.norm(x) ** 2

    def dJ_du(self, y, x):
        """
        Partial derivative of J with respect to the state y.
        """

        y_d = self.desired_state()
        return y - y_d

    def dJ_dx(self, y, x):
        """
        Partial derivative of J with respect to the control x.
        """

        return self.alpha * x

    def dc_dx_i(self, y, x, i):
        """
        Derivative of the state residual c(y, x) with respect to control x_i.

        The state equation is:

            A y - (B x + f) = 0

        so

            d c / d x_i = - B e_i

        where e_i is the i-th canonical basis vector.
        """

        e_i = np.zeros(self.num_dofs)
        e_i[i] = 1.0

        return -self._apply_control(e_i)

    def desired_state(self):
        """
        Desired state y_d used in the tracking objective.

        For the baseline experiment we use a manufactured target.
        You can change this later if needed.
        """

        return np.sin(np.pi * self.x)

    # =========================================================
    # BUILD OPERATOR A
    # =========================================================

    # def _build_operator(self):
    #     """
    #     1D finite difference discretization of:

    #         -(kappa(x) y')' + c(x) y

    #     For simplicity, this uses a standard tridiagonal stencil.
    #     """

    #     n = self.n
    #     h = self.h

    #     kappa = self._kappa(self.x)
    #     c = self._c(self.x)

    #     A = np.zeros((n, n))

    #     for i in range(n):

    #         # center coefficient
    #         A[i, i] = 2.0 * kappa[i] / h**2 + c[i]

    #         # left neighbor
    #         if i > 0:
    #             A[i, i - 1] = -kappa[i] / h**2

    #         # right neighbor
    #         if i < n - 1:
    #             A[i, i + 1] = -kappa[i] / h**2

    #     return A
    
    def _build_operator(self):
        """
        1D finite difference discretization of:

            -(kappa(x) y')' + c(x) y

        using a symmetric flux-form stencil so that the
        resulting matrix is Hermitian / symmetric.
        """

        n = self.n
        h = self.h

        kappa = self._kappa(self.x)
        c = self._c(self.x)

        A = np.zeros((n, n))

        # interface values kappa_{i+1/2}
        # simple arithmetic averaging
        kappa_half = np.zeros(n - 1)
        for i in range(n - 1):
            kappa_half[i] = 0.5 * (kappa[i] + kappa[i + 1])

        for i in range(n):

            # left interface kappa_{i-1/2}
            if i > 0:
                k_left = kappa_half[i - 1]
            else:
                # boundary interface near x=0
                k_left = kappa[i]

            # right interface kappa_{i+1/2}
            if i < n - 1:
                k_right = kappa_half[i]
            else:
                # boundary interface near x=1
                k_right = kappa[i]

            # diagonal
            A[i, i] = (k_left + k_right) / h**2 + c[i]

            # left neighbor
            if i > 0:
                A[i, i - 1] = -k_left / h**2

            # right neighbor
            if i < n - 1:
                A[i, i + 1] = -k_right / h**2

        return A

    # =========================================================
    # BUILD RHS
    # =========================================================

    def _build_rhs(self, u):
        """
        b = B u + f
        """

        Bu = self._apply_control(u)
        f = self._f(self.x)

        return Bu + f

    # =========================================================
    # CONTROL OPERATOR
    # =========================================================

    def _apply_control(self, u):
        """
        Apply B u

        exp4 → localized control
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
        Diffusion coefficient κ(x)
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
        Reaction term
        """

        if self.exp_type == "exp2":
            return np.ones_like(x)

        return np.zeros_like(x)

    # =========================================================
    # SOURCE TERM
    # =========================================================

    def _f(self, x):
        """
        Source term (exp5)
        """

        if self.exp_type == "exp5":
            return (np.pi ** 2) * np.sin(np.pi * x)

        return np.zeros_like(x)

    # =========================================================
    # CONTROL MASK
    # =========================================================

    def _control_mask(self, x):
        """
        Localized control ω = [0.2,0.4]
        """

        mask = np.zeros_like(x)
        mask[(x >= 0.2) & (x <= 0.4)] = 1.0

        return mask