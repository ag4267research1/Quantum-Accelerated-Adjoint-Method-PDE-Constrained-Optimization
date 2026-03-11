import numpy as np


def spectral_gradient(model, x, u, delta=1e-3, N=16, state_solver=None, **kwargs):
    """
    Estimate the gradient with respect to the control using the spectral method.

    This uses the spectral formula

        F(x) = (1 / (N delta)) sum_{k=0}^{N-1} omega^{-k} f(delta omega^k x)

    applied one coordinate at a time.

    Parameters
    ----------
    model : HeatModel
        PDE model.

    x : ndarray
        Current control vector.

    u : ndarray
        Current state vector.

    delta : float
        Small spectral step size.

    N : int
        Number of spectral sampling points.

    state_solver : callable
        State equation solver used to evaluate the reduced objective.

    Returns
    -------
    grad : ndarray
        Estimated gradient with respect to the control.
    """

    if state_solver is None:
        raise ValueError("state_solver must be provided to spectral_gradient")

    nx = len(x)
    grad = np.zeros(nx, dtype=float)

    omega = np.exp(-2j * np.pi / N)

    def reduced_objective(x_trial):
        """
        Reduced objective j(x) = J(u(x), x).
        """

        # Use the real part when solving the PDE, because the current
        # state solver is classical and expects real-valued controls.
        x_real = np.real(x_trial)

        u_trial = state_solver(model, x_real)
        return model.objective(u_trial, x_real)

    for i in range(nx):

        spectral_sum = 0.0 + 0.0j

        for k in range(N):

            step = delta * (omega ** k)

            # Create complex trial vector explicitly
            x_trial = np.array(x, dtype=complex)
            x_trial[i] += step

            value = reduced_objective(x_trial)

            spectral_sum += (omega ** (-k)) * value

        derivative = spectral_sum / (N * delta)

        grad[i] = np.real(derivative)

    return grad