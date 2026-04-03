import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.executer import Executer
from qlsas.post_processor import Post_Processor
from qlsas.transpiler import Transpiler

"""
Quantum adjoint solver wrapper for PDE-constrained optimization.

This project prepares the linear-system data needed for an
HHL + swap-test overlap query, instead of reconstructing the full adjoint
vector by measuring the solution register.

Main idea
---------
The optimizer expects the adjoint solve to return

    (p_state_like, p_scale)

where
    - p_state_like is something that the inner-product routine can use
    - p_scale = ||p|| is the norm of the recovered adjoint vector

In this swap-test-based version:
    - p_state_like is an AdjointSwapHandle
    - p_scale is computed classically from the original adjoint system

Two modes are supported:

1. No preconditioning
   We use the original adjoint system

       C_u^T p = J_u^T

   and prepare the padded HHL input directly.

2. Plain Jacobi preconditioning
   We apply left Jacobi preconditioning

       B         = D^{-1} C_u^T
       rhs_tilde = D^{-1} J_u^T

   where D = diag(C_u^T).

   Since plain left Jacobi generally destroys symmetry, we restore
   symmetry by embedding the preconditioned system into the block form

       H = [[0,   B ],
            [B^T, 0 ]]

   and prepare the embedded HHL input instead.

In both cases, the optimizer still gets:
    - a handle for overlap estimation through HHL + swap test
    - a scalar norm p_scale
"""

# ----------------------------------------------------------
# Data structure passed from adjoint_solver to inner_product
# ----------------------------------------------------------

@dataclass
class AdjointSwapHandle:
    """
    Lightweight container describing the HHL problem instance needed for a
    later swap-test overlap query.

    Fields
    ------
    system_matrix : ndarray
        The padded matrix actually passed to HHL.
    rhs_vector : ndarray
        The normalized padded right-hand side actually passed to HHL.
    original_dim : int
        The unpadded dimension of the system passed to HHL.
        This is n in the direct solve case, and 2n in the embedded case.
    state_dim : int
        The original adjoint dimension n.
    embedded : bool
        Whether the HHL system is the block-embedded one.
    shots : int
        Default shot count to use in the swap test path.
    """
    system_matrix: np.ndarray
    rhs_vector: np.ndarray
    original_dim: int
    state_dim: int
    embedded: bool
    shots: int


# ----------------------------------------------------------
# Small cache for HHL swap-test runtime objects
# ----------------------------------------------------------

_SWAP_RUNTIME_CACHE = {}


def _get_swap_runtime(dim):
    """
    Cache the HHL swap-test runtime objects keyed by padded dimension.
    """
    key = (dim, "swap_test")

    if key not in _SWAP_RUNTIME_CACHE:
        hhl = HHL(
            state_prep=StatePrep(method="default"),
            readout="swap_test",
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle="classical",
        )

        backend = AerSimulator()
        executer = Executer()
        post_processor = Post_Processor()

        _SWAP_RUNTIME_CACHE[key] = (hhl, backend, executer, post_processor)

    return _SWAP_RUNTIME_CACHE[key]


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def _next_power_of_two(n):
    """
    Return the smallest power of two greater than or equal to n.

    Example
    -------
    n = 5  -> returns 8
    n = 8  -> returns 8
    n = 0  -> returns 1
    """
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A, b):
    r"""
    Pad a square linear system \(A x = b\) to the next power-of-two dimension.

    Let \(n = \dim(b)\), and let \(m\) be the smallest power of two such that
    \(m \ge n\). We construct the padded system

    \[
    A_{\mathrm{pad}} x_{\mathrm{pad}} = b_{\mathrm{pad}},
    \]

    where

    \[
    A_{\mathrm{pad}} =
    \begin{bmatrix}
    A & 0 \\
    0 & I
    \end{bmatrix},
    \qquad
    b_{\mathrm{pad}} =
    \begin{bmatrix}
    b \\
    0
    \end{bmatrix}.
    \]

    The lower-right identity block prevents the padding from introducing
    singular directions.

    Parameters
    ----------
    A : ndarray
        Real square system matrix.
    b : ndarray
        Real right-hand side vector.

    Returns
    -------
    A_pad : ndarray
        Padded square matrix of power-of-two dimension.
    b_pad : ndarray
        Padded right-hand side vector.
    n : int
        Original unpadded dimension.
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


def _jacobi_left_precondition(C_u_T, J_u_T, eps=1e-12):
    """
    Apply plain Jacobi LEFT preconditioning to

        C_u^T p = J_u^T

    using

        B         = D^{-1} C_u^T
        rhs_tilde = D^{-1} J_u^T

    where D = diag(C_u^T).

    Notes
    -----
    Plain left Jacobi does NOT preserve symmetry by itself. That is why, when
    preconditioning is enabled, we later restore symmetry with a block embedding.
    """
    d = np.asarray(np.diag(C_u_T), dtype=float).copy()

    tiny = np.abs(d) < eps
    d[tiny] = eps

    D_inv = np.diag(1.0 / d)

    B = D_inv @ C_u_T
    rhs_tilde = D_inv @ J_u_T

    return B, rhs_tilde, D_inv


def _hermitian_embed(B):
    """
    Build the real symmetric block embedding

        H = [[0,  B ],
             [B^T, 0 ]].

    Since this pipeline is real-valued, Hermitian = symmetric.
    """
    n = B.shape[0]
    H = np.zeros((2 * n, 2 * n), dtype=float)
    H[:n, n:] = B
    H[n:, :n] = B.T
    return H


def _embedded_rhs(rhs):
    """
    Build the embedded right-hand side

        [rhs; 0]

    for the block system

        [[0,  B ],
         [B^T, 0 ]] [u; p] = [rhs; 0].

    In this system, the lower block of the solution corresponds to p.
    """
    rhs = np.asarray(rhs, dtype=float).flatten()
    zeros = np.zeros_like(rhs)
    return np.concatenate([rhs, zeros])


def _vector_to_state_and_norm(vec):
    """
    Convert a recovered real classical vector into

        (normalized Statevector, original norm).

    This is still used in the zero-RHS branch.
    """
    vec = np.asarray(vec, dtype=float).flatten()
    m = _next_power_of_two(len(vec))

    vec_pad = np.zeros(m, dtype=float)
    vec_pad[:len(vec)] = vec

    vec_norm = np.linalg.norm(vec_pad)

    if vec_norm == 0:
        zero_state = np.zeros(m, dtype=float)
        zero_state[0] = 1.0
        return Statevector(zero_state), 0.0

    vec_pad = vec_pad / vec_norm
    return Statevector(vec_pad), float(vec_norm)


def _classical_adjoint_norm(C_u_T, J_u_T):
    """
    Compute ||p|| from the original adjoint system

        C_u^T p = J_u^T

    using a classical solve. This provides the scale factor needed by
    the optimizer while the overlap itself is estimated through HHL + swap test.
    """
    try:
        p = np.linalg.solve(C_u_T, J_u_T)
    except np.linalg.LinAlgError:
        p = np.linalg.lstsq(C_u_T, J_u_T, rcond=None)[0]

    return float(np.linalg.norm(p))


def _classical_overlap_sign(system_matrix, rhs_vector, swap_test_vector):
    """
    Recover the sign of the inner product classically from the corresponding
    linear-system solve. This is used only for testing.
    """
    try:
        classical_solution = np.linalg.solve(system_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        classical_solution = np.linalg.lstsq(system_matrix, rhs_vector, rcond=None)[0]

    sign = np.sign(np.dot(swap_test_vector, classical_solution))
    if sign == 0:
        sign = 1.0
    return float(sign)


def _build_embedded_test_vector(handle, right):
    """
    Build the swap-test vector in the same padded space as the HHL solve.

    If the adjoint handle corresponds to the embedded system

        H [u; p] = [rhs; 0],

    then we compare against

        [0; w_i]

    so that the overlap targets the lower p block.

    If the handle corresponds to the direct system, we compare against w_i
    directly.

    Returns
    -------
    v_unit : ndarray
        Normalized padded swap-test vector.
    w_norm : float
        Norm of the unnormalized padded vector, used to restore scale.
    """
    w_vec = np.asarray(right, dtype=float).flatten()
    padded_dim = len(handle.rhs_vector)

    if handle.embedded:
        v = np.zeros(handle.original_dim, dtype=float)
        n = handle.state_dim
        v[n:n + len(w_vec)] = w_vec
    else:
        v = np.zeros(handle.original_dim, dtype=float)
        v[:len(w_vec)] = w_vec

    v_pad = np.zeros(padded_dim, dtype=float)
    v_pad[:len(v)] = v

    w_norm = np.linalg.norm(v_pad)
    if w_norm == 0:
        return v_pad, 0.0

    v_unit = v_pad / w_norm
    return v_unit, w_norm


# ----------------------------------------------------------
# Public API 1: adjoint solve preparation
# ----------------------------------------------------------

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
    r"""
    Prepare the real-valued adjoint system for HHL + swap-test overlap queries.

    Original adjoint system:

        C_u^T p = J_u^T.

    Behavior
    --------
    1. If use_preconditioning=False:
       prepare the usual padded system directly on C_u^T.

    2. If use_preconditioning=True:
       form plain Jacobi-left-preconditioned system

           B         = D^{-1} C_u^T
           rhs_tilde = D^{-1} J_u^T

       then restore symmetry by embedding it as

           H = [[0,  B ],
                [B^T, 0 ]]

       and prepare the embedded system

           H [u; p] = [rhs_tilde; 0].

    Returns
    -------
    (handle, p_scale)
        handle  : AdjointSwapHandle consumed later by inner_product
        p_scale : ||p|| from the original adjoint system

    or

    (handle, p_scale, diagnostics) if return_diagnostics=True
    """

    # --------------------------------------------------
    # Form adjoint operator and RHS
    # --------------------------------------------------
    C_u = np.asarray(A, dtype=float).copy()
    J_u_T = np.asarray(rhs, dtype=float).copy()
    C_u_T = C_u.T

    cond_raw = np.linalg.cond(C_u_T)

    # CHANGED: compute p_scale classically from the original adjoint system
    p_scale = _classical_adjoint_norm(C_u_T, J_u_T)

    # --------------------------------------------------
    # Zero-RHS shortcut
    # --------------------------------------------------
    rhs_norm_direct = np.linalg.norm(J_u_T)
    if rhs_norm_direct == 0:
        p_state, p_scale_zero = _vector_to_state_and_norm(np.zeros_like(J_u_T))
        if return_diagnostics:
            return p_state, p_scale_zero, {
                "cond_raw": cond_raw,
                "cond_pre": cond_raw,
                "cond_pad": cond_raw,
                "rhs_norm": 0.0,
                "used_preconditioning": False,
            }
        return p_state, p_scale_zero

    # --------------------------------------------------
    # Case 1: No preconditioning -> direct padded HHL data
    # --------------------------------------------------
    if not use_preconditioning:
        if check_hermitian and not np.allclose(C_u_T, C_u_T.T, atol=1e-10):
            raise ValueError("Adjoint matrix passed to HHL is not symmetric.")

        b_unit = J_u_T / rhs_norm_direct

        A_pad, b_pad, original_dim = _pad_linear_system(C_u_T, b_unit)
        cond_pad = np.linalg.cond(A_pad)

        handle = AdjointSwapHandle(
            system_matrix=A_pad,
            rhs_vector=b_pad,
            original_dim=original_dim,
            state_dim=len(J_u_T),
            embedded=False,
            shots=int(shots),
        )

        if return_diagnostics:
            return handle, p_scale, {
                "cond_raw": cond_raw,
                "cond_pre": cond_raw,
                "cond_pad": cond_pad,
                "rhs_norm": rhs_norm_direct,
                "used_preconditioning": False,
            }

        return handle, p_scale

    # --------------------------------------------------
    # Case 2: Plain Jacobi preconditioning + block embedding
    # --------------------------------------------------
    B, rhs_tilde, D_inv = _jacobi_left_precondition(C_u_T, J_u_T, eps=eps)
    cond_pre = np.linalg.cond(B)

    H = _hermitian_embed(B)

    if check_hermitian and not np.allclose(H, H.T, atol=1e-10):
        raise ValueError("Embedded preconditioned matrix passed to HHL is not symmetric.")

    b_embed = _embedded_rhs(rhs_tilde)

    rhs_norm = np.linalg.norm(b_embed)
    if rhs_norm == 0:
        p_state, p_scale_zero = _vector_to_state_and_norm(np.zeros_like(J_u_T))
        if return_diagnostics:
            return p_state, p_scale_zero, {
                "cond_raw": cond_raw,
                "cond_pre": cond_pre,
                "cond_pad": cond_pre,
                "rhs_norm": 0.0,
                "embedded_dim": 2 * len(J_u_T),
                "used_preconditioning": True,
            }
        return p_state, p_scale_zero

    b_unit = b_embed / rhs_norm

    H_pad, b_pad, original_dim = _pad_linear_system(H, b_unit)
    cond_pad = np.linalg.cond(H_pad)

    handle = AdjointSwapHandle(
        system_matrix=H_pad,
        rhs_vector=b_pad,
        original_dim=original_dim,
        state_dim=len(J_u_T),
        embedded=True,
        shots=int(shots),
    )

    if return_diagnostics:
        return handle, p_scale, {
            "cond_raw": cond_raw,
            "cond_pre": cond_pre,
            "cond_pad": cond_pad,
            "rhs_norm": rhs_norm,
            "embedded_dim": 2 * len(J_u_T),
            "used_preconditioning": True,
        }

    return handle, p_scale


# ----------------------------------------------------------
# Public API 2: overlap estimation
# ----------------------------------------------------------

def inner_product(left, right, shots=1024, **kwargs):
    r"""
    Estimate the overlap magnitude used in the reduced-gradient assembly.
    """

    # --------------------------------------------------
    # Mode 1: HHL + swap-test readout from adjoint handle
    # --------------------------------------------------
    if isinstance(left, AdjointSwapHandle):
        v_unit, w_norm = _build_embedded_test_vector(left, right)

        if w_norm == 0:
            return 0.0

        dim = len(left.rhs_vector)
        hhl, backend, executer, post_processor = _get_swap_runtime(dim)

        circuit = hhl.build_circuit(
            left.system_matrix,
            left.rhs_vector,
            swap_test_vector=v_unit,
        )

        transpiler = Transpiler(
            circuit=circuit,
            backend=backend,
            optimization_level=0,
        )
        transpiled_circuit = transpiler.optimize()

        result = executer.run(
            transpiled_circuit,
            backend,
            int(shots),
            verbose=False,
        )

        # CHANGED: process_swap_test(...)[0] is P(1 | success), not the overlap.
        exp_value = post_processor.process_swap_test(
            result,
            left.system_matrix,
            left.rhs_vector,
            v_unit,
        )[0]

        # For the swap test:
        #   P(1) = (1 - |<v|x>|^2) / 2
        # so
        #   |<v|x>| = sqrt(max(0, 1 - 2 P(1))).
        overlap_mag = np.sqrt(max(0.0, 1.0 - 2.0 * exp_value))

        # CHANGED: recover the sign classically for testing
        sign = _classical_overlap_sign(
            left.system_matrix,
            left.rhs_vector,
            v_unit,
        )

        # Restore only ||w_i|| here.
        return float(sign * overlap_mag * w_norm)

    # --------------------------------------------------
    # Mode 2: fallback local standalone swap test
    # --------------------------------------------------
    if isinstance(left, Statevector):
        p_vec = np.asarray(np.real(left.data), dtype=float)
        left_is_statevector = True
    else:
        p_vec = np.asarray(left, dtype=float)
        left_is_statevector = False

    w_vec = np.asarray(right, dtype=float)

    max_len = max(len(p_vec), len(w_vec))
    n = max(1, int(np.ceil(np.log2(max_len))))
    size = 2 ** n

    p_pad = np.zeros(size, dtype=float)
    w_pad = np.zeros(size, dtype=float)

    p_pad[:len(p_vec)] = p_vec
    w_pad[:len(w_vec)] = w_vec

    p_norm = np.linalg.norm(p_pad)
    w_norm = np.linalg.norm(w_pad)

    if p_norm == 0 or w_norm == 0:
        return 0.0

    p_pad = p_pad / p_norm
    w_pad = w_pad / w_norm

    qc = QuantumCircuit(1 + 2 * n, 1)

    qc.initialize(p_pad, range(1, n + 1))
    qc.initialize(w_pad, range(n + 1, 2 * n + 1))

    qc.h(0)

    for i in range(n):
        qc.cswap(0, 1 + i, 1 + n + i)

    qc.h(0)
    qc.measure(0, 0)

    backend = AerSimulator()
    result = backend.run(qc, shots=int(shots)).result()
    counts = result.get_counts()

    p0 = counts.get("0", 0) / int(shots)
    overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

    # CHANGED: recover sign classically for testing in the fallback path too
    sign = np.sign(np.dot(p_pad, w_pad))
    if sign == 0:
        sign = 1.0

    if left_is_statevector:
        overlap = w_norm * overlap
    else:
        overlap = p_norm * w_norm * overlap

    return float(sign * overlap)