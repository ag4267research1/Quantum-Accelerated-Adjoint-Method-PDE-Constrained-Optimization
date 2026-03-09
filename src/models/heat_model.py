import numpy as np


class HeatModel:
    """
    1D steady heat equation with radiation and convection

    d²T/dy² = ε(T)(T^4 − T_inf^4) + h(T − T_inf)

    Domain: y ∈ [0,1]
    Boundary conditions:
        T(0) = 0
        T(1) = 0
    """

    def __init__(self, n=100, T_inf=50.0, h=0.5):
        self.n = n
        self.T_inf = T_inf
        self.h = h

        self.dy = 1.0 / (n - 1)

        self.y = np.linspace(0, 1, n)

    # ---------------------------------------------
    # Radiation coefficient ε(T)
    # ---------------------------------------------
    def epsilon(self, T):

        return (1 + 5 * np.sin(3 * np.pi * T / 200)
                + np.exp(0.02 * T)) * 1e-4

    # ---------------------------------------------
    # PDE residual c(x,u)
    # ---------------------------------------------
    def residual(self, T):

        r = np.zeros_like(T)

        dy2 = self.dy ** 2

        for i in range(1, self.n - 1):

            laplacian = (T[i+1] - 2*T[i] + T[i-1]) / dy2

            radiation = self.epsilon(T[i]) * (T[i]**4 - self.T_inf**4)

            convection = self.h * (T[i] - self.T_inf)

            r[i] = laplacian - radiation - convection

        # boundary conditions
        r[0] = T[0]
        r[-1] = T[-1]

        return r

    # ---------------------------------------------
    # Jacobian ∂c/∂T
    # ---------------------------------------------
    def jacobian(self, T):

        n = self.n
        dy2 = self.dy ** 2

        A = np.zeros((n, n))

        for i in range(1, n-1):

            A[i, i-1] = 1/dy2
            A[i, i] = -2/dy2
            A[i, i+1] = 1/dy2

        # boundary rows
        A[0,0] = 1
        A[-1,-1] = 1

        return A

    # ---------------------------------------------
    # Objective function
    # J(x,u)
    # ---------------------------------------------
    def objective(self, T, target=None):

        if target is None:
            target = np.zeros_like(T)

        return 0.5 * np.linalg.norm(T - target) ** 2

    # ---------------------------------------------
    # ∂J/∂x
    # ---------------------------------------------
    def dJ_dx(self, T, target=None):

        if target is None:
            target = np.zeros_like(T)

        return T - target

    # ---------------------------------------------
    # initial guess
    # ---------------------------------------------
    def initial_state(self):

        return np.zeros(self.n)