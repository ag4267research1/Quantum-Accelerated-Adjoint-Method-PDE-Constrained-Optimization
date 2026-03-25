import numpy as np

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.solver import QuantumLinearSolver
from qiskit_ibm_runtime import QiskitRuntimeService


# ----------------------------------------------------------
# Global caches
# ----------------------------------------------------------

_SOLVER_CACHE = {}
_SERVICE_CACHE = None
_BACKEND_CACHE = {}


def _next_power_of_two(n):
    """Return smallest power of two >= n."""
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A, b):
    """
    Pad Ax=b so the dimension becomes a power of two.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n = len(b)
    m = _next_power_of_two(n)

    if m == n:
        return A, b, n

    A_pad = np.eye(m, dtype=float)
    A_pad[:n, :n] = A

    b_pad = np.zeros(m, dtype=float)
    b_pad[:n] = b

    return A_pad, b_pad, n


def _get_runtime_service():
    """
    Return a cached IBM Runtime service object.
    """
    global _SERVICE_CACHE

    if _SERVICE_CACHE is None:
        _SERVICE_CACHE = QiskitRuntimeService()

    return _SERVICE_CACHE


def _get_backend(backend_name=None):
    """
    Return a cached IBM backend.

    If backend_name is None, choose the least busy operational
    non-simulator backend.
    """
    key = backend_name

    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    service = _get_runtime_service()

    if backend_name is not None:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)

    _BACKEND_CACHE[key] = backend
    return backend


def _get_solver(dim, shots, backend_name=None):
    """
    Cache the solver so we do not rebuild the HHL/solver wrapper
    every optimizer iteration.
    """
    key = (dim, shots, backend_name)

    if key not in _SOLVER_CACHE:
        hhl = HHL(
            state_prep=StatePrep(method="default"),
            readout="measure_x",
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle="classical",
        )

        backend = _get_backend(backend_name=backend_name)

        solver = QuantumLinearSolver(
            qlsa=hhl,
            backend=backend,
            shots=shots,
            optimization_level=0,
        )

        _SOLVER_CACHE[key] = solver

    return _SOLVER_CACHE[key]


def adjoint_solver(A, rhs, shots=64, backend_name=None, **kwargs):
    """
    Solve the adjoint system

        A^T p = rhs

    using the QLSA/HHL pipeline on an IBM backend.

    Parameters
    ----------
    A : ndarray
        System / Jacobian matrix.

    rhs : ndarray
        Right-hand side.

    shots : int
        Number of shots used by the backend.

    backend_name : str or None
        Specific IBM backend name. If None, choose least busy.

    Returns
    -------
    ndarray
        Approximate adjoint vector in the original unpadded dimension.
    """

    # --------------------------------------------------
    # Clean inputs
    # --------------------------------------------------

    A = np.asarray(A, dtype=float)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)

    # --------------------------------------------------
    # Form adjoint system
    # --------------------------------------------------

    AT = A.T

    # --------------------------------------------------
    # Normalize RHS for state preparation
    # --------------------------------------------------

    rhs_norm = np.linalg.norm(rhs)

    if rhs_norm == 0:
        return np.zeros_like(rhs)

    b = rhs / rhs_norm

    # --------------------------------------------------
    # Pad system
    # --------------------------------------------------

    AT_pad, b_pad, original_dim = _pad_linear_system(AT, b)
    dim = len(b_pad)

    # --------------------------------------------------
    # Get cached solver
    # --------------------------------------------------

    solver = _get_solver(
        dim=dim,
        shots=shots,
        backend_name=backend_name,
    )

    # --------------------------------------------------
    # Solve using HHL framework
    # --------------------------------------------------

    solution = solver.solve(AT_pad, b_pad)

    # --------------------------------------------------
    # Remove padding
    # --------------------------------------------------

    p = np.asarray(solution, dtype=float).reshape(-1)[:original_dim]

    # --------------------------------------------------
    # Keep normalized state-like output for swap-test workflow
    # --------------------------------------------------

    return p