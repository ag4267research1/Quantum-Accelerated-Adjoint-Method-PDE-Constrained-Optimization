import numpy as np

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.solver import QuantumLinearSolver
from qiskit.quantum_info import Statevector   # CHANGED: return a Statevector for swap test
from qiskit_aer import AerSimulator


# ----------------------------------------------------------
# Global cache
# ----------------------------------------------------------

_SOLVER_CACHE = {}


def _next_power_of_two(n):
    """Return smallest power of two >= n."""
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A, b):
    """Pad Ax=b so the dimension becomes a power of two."""
    n = len(b)
    m = _next_power_of_two(n)

    if m == n:
        return A, b, n

    A_pad = np.eye(m, dtype=float)
    A_pad[:n, :n] = A

    b_pad = np.zeros(m, dtype=float)
    b_pad[:n] = b

    return A_pad, b_pad, n


def _get_solver(dim, shots):
    """
    Cache the solver so we don't rebuild/transpile HHL
    every optimizer iteration.
    """
    key = (dim, shots)

    if key not in _SOLVER_CACHE:
        hhl = HHL(
            state_prep=StatePrep(method="default"),
            readout="measure_x",
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle="classical",
        )

        backend = AerSimulator()

        solver = QuantumLinearSolver(
            qlsa=hhl,
            backend=backend,
            shots=shots,
            optimization_level=0,
        )

        _SOLVER_CACHE[key] = solver

    return _SOLVER_CACHE[key]


def _hermitianize(A):
    """
    Numerical cleanup to enforce exact Hermitian symmetry.
    For real matrices this is just symmetrization.
    """
    return 0.5 * (A + A.T.conjugate())


def _symmetric_jacobi_precondition(A, rhs, eps=1e-12):
    """
    Apply symmetric Jacobi scaling

        A_tilde = D^{-1/2} A D^{-1/2}
        b_tilde = D^{-1/2} rhs

    which preserves Hermitian structure.

    Returns
    -------
    A_tilde, b_tilde, D_inv_sqrt
    """
    d = np.real(np.diag(A)).copy()

    # If the full diagonal is negative, flip the whole system sign.
    if np.all(d < 0):
        A = -A
        rhs = -rhs
        d = -d

    # Guard against zero/tiny diagonal entries.
    d[np.abs(d) < eps] = eps

    if np.any(d <= 0):
        raise ValueError(
            "Symmetric Jacobi requires a positive diagonal. "
            "The adjoint matrix diagonal is not suitable for D^{-1/2} scaling."
        )

    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))

    A_tilde = D_inv_sqrt @ A @ D_inv_sqrt
    b_tilde = D_inv_sqrt @ rhs

    # Remove tiny numerical asymmetry
    A_tilde = _hermitianize(A_tilde)

    return A_tilde, b_tilde, D_inv_sqrt


def _vector_to_state_and_norm(vec):
    """
    CHANGED: convert a recovered classical vector into
    (normalized Statevector, original norm) for the swap test.
    """
    vec = np.asarray(vec, dtype=complex).flatten()
    m = _next_power_of_two(len(vec))

    vec_pad = np.zeros(m, dtype=complex)
    vec_pad[:len(vec)] = vec

    vec_norm = np.linalg.norm(vec_pad)

    if vec_norm == 0:
        zero_state = np.zeros(m, dtype=complex)
        zero_state[0] = 1.0
        return Statevector(zero_state), 0.0

    vec_pad = vec_pad / vec_norm
    return Statevector(vec_pad), float(vec_norm)


def adjoint_solver(
    A,
    rhs,
    shots=2048,
    use_preconditioning=False,
    eps=1e-12,
    check_hermitian=True,
    return_diagnostics=False,
    **kwargs,
):
    """
    Solve the adjoint system

        A^T p = rhs

    using HHL. Optionally applies symmetric Jacobi preconditioning
    that preserves Hermitian structure.

    Parameters
    ----------
    A : ndarray
        State Jacobian.
    rhs : ndarray
        Right-hand side of adjoint system.
    shots : int
        Number of shots for the quantum backend.
    use_preconditioning : bool
        Whether to apply symmetric Jacobi preconditioning.
    eps : float
        Small diagonal safeguard for preconditioning.
    check_hermitian : bool
        Whether to enforce Hermitian checks.
    return_diagnostics : bool
        If True, also return a dict with conditioning info.

    Returns
    -------
    (p_state, p_scale) : tuple
        p_state is a normalized Statevector for the swap test.
        p_scale = ||p|| is the recovered adjoint norm in the original coordinates.

    or

    ((p_state, p_scale), diagnostics) if return_diagnostics=True
    """

    # --------------------------------------------------
    # Form adjoint system
    # --------------------------------------------------
    A_adj = np.asarray(A.T, dtype=float).copy()
    rhs_adj = np.asarray(rhs, dtype=float).copy()

    # Numerical Hermitian cleanup
    A_adj = _hermitianize(A_adj)

    if check_hermitian and not np.allclose(A_adj, A_adj.T.conjugate(), atol=1e-10):
        raise ValueError("Adjoint matrix is not Hermitian.")

    cond_raw = np.linalg.cond(A_adj)

    # --------------------------------------------------
    # Optional preconditioning
    # --------------------------------------------------
    if use_preconditioning:
        A_hhl, rhs_hhl, D_inv_sqrt = _symmetric_jacobi_precondition(
            A_adj, rhs_adj, eps=eps
        )
    else:
        A_hhl = A_adj
        rhs_hhl = rhs_adj
        D_inv_sqrt = None

    if check_hermitian and not np.allclose(A_hhl, A_hhl.T.conjugate(), atol=1e-10):
        raise ValueError("Matrix passed to HHL is not Hermitian.")

    cond_pre = np.linalg.cond(A_hhl)

    # --------------------------------------------------
    # Normalize RHS for HHL
    # --------------------------------------------------
    rhs_norm = np.linalg.norm(rhs_hhl)
    if rhs_norm == 0:
        # CHANGED: return normalized quantum state + zero scale
        p_state, p_scale = _vector_to_state_and_norm(np.zeros_like(rhs_adj))
        if return_diagnostics:
            return (p_state, p_scale), {
                "cond_raw": cond_raw,
                "cond_pre": cond_pre,
                "cond_pad": cond_pre,
                "rhs_norm": 0.0,
            }
        return p_state, p_scale

    b_unit = rhs_hhl / rhs_norm

    # --------------------------------------------------
    # Pad system
    # --------------------------------------------------
    A_pad, b_pad, original_dim = _pad_linear_system(A_hhl, b_unit)
    cond_pad = np.linalg.cond(A_pad)
    dim = len(b_pad)

    # --------------------------------------------------
    # Solve using cached HHL framework
    # --------------------------------------------------
    solver = _get_solver(dim, shots)
    solution = solver.solve(A_pad, b_pad)

    # --------------------------------------------------
    # Remove padding
    # --------------------------------------------------
    y = np.asarray(solution[:original_dim], dtype=float)

    # --------------------------------------------------
    # Undo RHS normalization
    # --------------------------------------------------
    y = y * rhs_norm

    # --------------------------------------------------
    # Recover original adjoint variable
    # --------------------------------------------------
    if D_inv_sqrt is not None:
        p = D_inv_sqrt @ y
    else:
        p = y

    # CHANGED: return normalized quantum state + recovered norm
    p_state, p_scale = _vector_to_state_and_norm(p)

    if return_diagnostics:
        return (p_state, p_scale), {
            "cond_raw": cond_raw,
            "cond_pre": cond_pre,
            "cond_pad": cond_pad,
            "rhs_norm": rhs_norm,
        }

    return p_state, p_scale