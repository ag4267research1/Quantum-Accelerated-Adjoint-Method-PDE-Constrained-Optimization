import numpy as np

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.solver import QuantumLinearSolver
from qiskit_aer import AerSimulator


def _next_power_of_two(n):
    """
    Return the next power of two greater than or equal to n.

    Parameters
    ----------
    n : int
        Input dimension.

    Returns
    -------
    int
        Smallest power of two >= n.
    """

    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A, b):
    """
    Pad a linear system Ax = b so that its dimension is a power of two.

    The QLSAs implementation requires state vectors to have length 2^m.
    If the input system dimension is not a power of two, this function
    embeds the system into a larger padded system.

    Parameters
    ----------
    A : ndarray
        Square matrix of shape (n, n).

    b : ndarray
        Right-hand side vector of length n.

    Returns
    -------
    A_pad : ndarray
        Padded matrix of shape (m, m), where m is a power of two.

    b_pad : ndarray
        Padded vector of length m.

    original_dim : int
        Original system dimension n.
    """

    n = len(b)
    m = _next_power_of_two(n)

    if m == n:
        return A, b, n

    A_pad = np.eye(m, dtype=float)
    A_pad[:n, :n] = A

    b_pad = np.zeros(m, dtype=float)
    b_pad[:n] = b

    return A_pad, b_pad, n


def adjoint_solver(A, rhs, shots=1000, **kwargs):
    """
    Solve the adjoint system using a quantum linear system algorithm.

        A^T p = rhs

    The QLSAs package expects the state dimension to be a power of two,
    so the system is padded automatically when needed.

    Parameters
    ----------
    A : ndarray
        Jacobian matrix ∂c/∂u.

    rhs : ndarray
        Right-hand side vector ∂J/∂u.

    shots : int
        Number of successful measurement shots requested from the solver.

    Returns
    -------
    p : ndarray
        Classical approximation of the adjoint vector, truncated back
        to the original dimension.
    """

    # ---------------------------------------------
    # Form the adjoint system A^T p = rhs
    # ---------------------------------------------
    AT = A.T

    # ---------------------------------------------
    # Normalize the right-hand side
    # ---------------------------------------------
    rhs_norm = np.linalg.norm(rhs)

    if rhs_norm == 0.0:
        return np.zeros_like(rhs)

    b = rhs / rhs_norm

    # ---------------------------------------------
    # Pad system to power-of-two dimension
    # ---------------------------------------------
    AT_pad, b_pad, original_dim = _pad_linear_system(AT, b)

    # ---------------------------------------------
    # Build HHL object
    # ---------------------------------------------
    hhl = HHL(
        state_prep=StatePrep(method="default"),
        readout="measure_x",
        num_qpe_qubits=int(np.log2(len(b_pad))),
        eig_oracle="classical"
    )

    # ---------------------------------------------
    # Use local simulator backend
    # ---------------------------------------------
    backend = AerSimulator()

    solver = QuantumLinearSolver(
        qlsa=hhl,
        backend=backend,
        target_successful_shots=shots,
        shots_per_batch=shots,
        optimization_level=3
    )

    # ---------------------------------------------
    # Solve padded system
    # ---------------------------------------------
    solution = solver.solve(AT_pad, b_pad)

    # ---------------------------------------------
    # Extract quantum state |p>
    # ---------------------------------------------
    if isinstance(solution, dict):
        p_state = solution["solution"]
    else:
        p_state = solution

    # ---------------------------------------------
    # Return quantum state and scaling factor
    # ---------------------------------------------
    return p_state, rhs_norm