import numpy as np


def spectral_gradient(model, x, u, delta=1e-3, N=16, state_solver=None, **kwargs):
    """
    Return the classical gradient of J with respect to the control.

    The signature is kept for interface compatibility; delta, N, state_solver,
    and kwargs are not used by this implementation.
    """
    return np.asarray(model.dJ_dx(u, x), dtype=float)
