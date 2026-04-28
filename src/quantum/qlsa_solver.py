import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.state_prep import DefaultStatePrep
from qlsas.executer import Executer
from qlsas.post_processor import Post_Processor
from qlsas.transpiler import Transpiler
from qlsas.algorithms.hhl.eig_oracles import ClassicalEigOracle
from qlsas.readout.swap_test import SwapTestReadout
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

This version uses ONLY the direct adjoint system

    C_u^T p = J_u^T

with power-of-two padding. Preconditioning has been removed completely.

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
    state_dim : int
        The original adjoint dimension n.
    shots : int
        Default shot count to use in the swap test path.
    p_scale : float
        Classical norm of the adjoint vector from the original adjoint system.
    """
    system_matrix: np.ndarray
    rhs_vector: np.ndarray
    original_dim: int
    state_dim: int
    shots: int
    p_scale: float


# ----------------------------------------------------------
# Small cache for HHL swap-test runtime objects
# ----------------------------------------------------------

_SWAP_RUNTIME_CACHE = {}


def _build_backend(
    backend_mode="aer",
    ibm_backend_name=None,
    ibm_channel=None,
    ibm_token=None,
    ibm_instance=None,
    ibm_use_least_busy=False,
):
    """
    Build either a local Aer simulator backend or an IBM backend.

    Parameters
    ----------
    backend_mode : str
        "aer" or "ibm".

    ibm_backend_name : str or None
        Specific IBM backend name, e.g. "ibm_kingston".

    ibm_channel : str or None
        Optional channel passed to QiskitRuntimeService.

    ibm_token : str or None
        Optional IBM API token. If omitted, the saved/default account is used.

    ibm_instance : str or None
        Optional IBM instance / CRN.

    ibm_use_least_busy : bool
        If True and no explicit backend name is provided, choose the least-busy
        operational real backend.
    """
    if backend_mode == "aer":
        return AerSimulator()

    if backend_mode == "ibm":
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise ImportError(
                "qiskit-ibm-runtime is required for backend_mode='ibm'. "
                "Install it with: pip install qiskit-ibm-runtime"
            ) from exc

        service_kwargs = {}
        if ibm_channel is not None:
            service_kwargs["channel"] = ibm_channel
        if ibm_token is not None:
            service_kwargs["token"] = ibm_token
        if ibm_instance is not None:
            service_kwargs["instance"] = ibm_instance

        service = QiskitRuntimeService(**service_kwargs)

        if ibm_backend_name is not None:
            return service.backend(ibm_backend_name, instance=ibm_instance)

        if ibm_use_least_busy:
            return service.least_busy(
                operational=True,
                simulator=False,
                instance=ibm_instance,
            )

        raise ValueError(
            "For backend_mode='ibm', provide ibm_backend_name or set "
            "ibm_use_least_busy=True."
        )

    raise ValueError(f"Unknown backend_mode: {backend_mode}")


def _get_swap_runtime(
    dim,
    backend_mode="aer",
    ibm_backend_name=None,
    ibm_channel=None,
    ibm_token=None,
    ibm_instance=None,
    ibm_use_least_busy=False,
):
    """
    Cache the HHL swap-test runtime objects keyed by padded dimension and backend.
    """
    key = (
        dim,
        backend_mode,
        ibm_backend_name,
        ibm_channel,
        ibm_instance,
        ibm_use_least_busy,
    )

    if key not in _SWAP_RUNTIME_CACHE:
        hhl = HHL(
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle=ClassicalEigOracle(),
        )

        backend = _build_backend(
            backend_mode=backend_mode,
            ibm_backend_name=ibm_backend_name,
            ibm_channel=ibm_channel,
            ibm_token=ibm_token,
            ibm_instance=ibm_instance,
            ibm_use_least_busy=ibm_use_least_busy,
        )

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


def _build_test_vector(handle, right):
    """
    Build the swap-test vector in the same padded space as the HHL solve.

    Since preconditioning/embedding has been removed, we compare directly
    against w_i in the padded HHL space.

    Returns
    -------
    v_unit : ndarray
        Normalized padded swap-test vector.
    w_norm : float
        Norm of the unnormalized padded vector, used to restore scale.
    """
    w_vec = np.asarray(right, dtype=float).flatten()
    padded_dim = len(handle.rhs_vector)

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

    Notes
    -----
    The `use_preconditioning` argument is kept only for interface
    compatibility with the rest of the codebase, but preconditioning
    is not used in this version.
    """

    # --------------------------------------------------
    # Form adjoint operator and RHS
    # --------------------------------------------------
    C_u = np.asarray(A, dtype=float).copy()
    J_u_T = np.asarray(rhs, dtype=float).copy()
    C_u_T = C_u.T

    cond_raw = np.linalg.cond(C_u_T)

    # compute p_scale classically from the original adjoint system
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
                "cond_pad": cond_raw,
                "rhs_norm": 0.0,
                "used_preconditioning": False,
            }
        return p_state, p_scale_zero

    # --------------------------------------------------
    # Direct padded HHL data only
    # --------------------------------------------------
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
        shots=int(shots),
        p_scale=float(p_scale),
    )

    if return_diagnostics:
        return handle, p_scale, {
            "cond_raw": cond_raw,
            "cond_pad": cond_pad,
            "rhs_norm": rhs_norm_direct,
            "used_preconditioning": False,
        }

    return handle, p_scale


# ----------------------------------------------------------
# Public API 2: overlap estimation
# ----------------------------------------------------------
### Old version of QCOL QLSA 
# def inner_product(left, right, shots=1024, **kwargs):
#     r"""
#     Estimate the overlap magnitude used in the reduced-gradient assembly.
#     """

#     # --------------------------------------------------
#     # Mode 1: HHL + swap-test readout from adjoint handle
#     # --------------------------------------------------
#     if isinstance(left, AdjointSwapHandle):
#         v_unit, w_norm = _build_test_vector(left, right)

#         if w_norm == 0:
#             return 0.0

#         dim = len(left.rhs_vector)
#         hhl, backend, executer, post_processor = _get_swap_runtime(
#             dim,
#             backend_mode=kwargs.get("backend_mode", "aer"),
#             ibm_backend_name=kwargs.get("ibm_backend_name"),
#             ibm_channel=kwargs.get("ibm_channel"),
#             ibm_token=kwargs.get("ibm_token"),
#             ibm_instance=kwargs.get("ibm_instance"),
#             ibm_use_least_busy=kwargs.get("ibm_use_least_busy", False),
#         )

#         circuit = hhl.build_circuit(
#             left.system_matrix,
#             left.rhs_vector,
#             swap_test_vector=v_unit,
#         )

#         transpiler = Transpiler(
#             circuit=circuit,
#             backend=backend,
#             optimization_level=0,
#         )
#         transpiled_circuit = transpiler.optimize()

#         result = executer.run(
#             transpiled_circuit,
#             backend,
#             int(shots),
#             verbose=False,
#         )

#         # process_swap_test(...)[0] is P(1 | success), not the overlap.
#         exp_value = post_processor.process_swap_test(
#             result,
#             left.system_matrix,
#             left.rhs_vector,
#             v_unit,
#         )[0]

#         # For the swap test:
#         #   P(1) = (1 - |<v|x>|^2) / 2
#         # so
#         #   |<v|x>| = sqrt(max(0, 1 - 2 P(1))).
#         overlap_mag = np.sqrt(max(0.0, 1.0 - 2.0 * exp_value))

#         # recover the sign classically for testing
#         sign = _classical_overlap_sign(
#             left.system_matrix,
#             left.rhs_vector,
#             v_unit,
#         )

#         return float(sign * overlap_mag * w_norm)

#     # --------------------------------------------------
#     # Mode 2: fallback local standalone swap test
#     # --------------------------------------------------
#     if isinstance(left, Statevector):
#         p_vec = np.asarray(np.real(left.data), dtype=float)
#         left_is_statevector = True
#     else:
#         p_vec = np.asarray(left, dtype=float)
#         left_is_statevector = False

#     w_vec = np.asarray(right, dtype=float)

#     max_len = max(len(p_vec), len(w_vec))
#     n = max(1, int(np.ceil(np.log2(max_len))))
#     size = 2 ** n

#     p_pad = np.zeros(size, dtype=float)
#     w_pad = np.zeros(size, dtype=float)

#     p_pad[:len(p_vec)] = p_vec
#     w_pad[:len(w_vec)] = w_vec

#     p_norm = np.linalg.norm(p_pad)
#     w_norm = np.linalg.norm(w_pad)

#     if p_norm == 0 or w_norm == 0:
#         return 0.0

#     p_pad = p_pad / p_norm
#     w_pad = w_pad / w_norm

#     qc = QuantumCircuit(1 + 2 * n, 1)

#     qc.initialize(p_pad, range(1, n + 1))
#     qc.initialize(w_pad, range(n + 1, 2 * n + 1))

#     qc.h(0)

#     for i in range(n):
#         qc.cswap(0, 1 + i, 1 + n + i)

#     qc.h(0)
#     qc.measure(0, 0)

#     backend = _build_backend(
#         backend_mode=kwargs.get("backend_mode", "aer"),
#         ibm_backend_name=kwargs.get("ibm_backend_name"),
#         ibm_channel=kwargs.get("ibm_channel"),
#         ibm_token=kwargs.get("ibm_token"),
#         ibm_instance=kwargs.get("ibm_instance"),
#         ibm_use_least_busy=kwargs.get("ibm_use_least_busy", False),
#     )

#     result = backend.run(qc, shots=int(shots)).result()
#     counts = result.get_counts()

#     p0 = counts.get("0", 0) / int(shots)
#     overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

#     sign = np.sign(np.dot(p_pad, w_pad))
#     if sign == 0:
#         sign = 1.0

#     if left_is_statevector:
#         overlap = w_norm * overlap
#     else:
#         overlap = p_norm * w_norm * overlap

#     return float(sign * overlap)


#Latest QCOL QLSA refactored

_SWAP_RUNTIME_CACHE = {}


def _get_swap_runtime(
    dim,
    backend_mode="aer",
    ibm_backend_name=None,
    ibm_channel=None,
    ibm_token=None,
    ibm_instance=None,
    ibm_use_least_busy=False,
):
    key = (
        int(dim),
        backend_mode,
        ibm_backend_name,
        ibm_channel,
        ibm_instance,
        bool(ibm_use_least_busy),
    )

    if key not in _SWAP_RUNTIME_CACHE:
        hhl = HHL(
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle=ClassicalEigOracle(),
        )

        backend = _build_backend(
            backend_mode=backend_mode,
            ibm_backend_name=ibm_backend_name,
            ibm_channel=ibm_channel,
            ibm_token=ibm_token,
            ibm_instance=ibm_instance,
            ibm_use_least_busy=ibm_use_least_busy,
        )

        executer = Executer()
        post_processor = Post_Processor()
        state_prep = DefaultStatePrep()

        _SWAP_RUNTIME_CACHE[key] = (
            hhl,
            backend,
            executer,
            post_processor,
            state_prep,
        )

    return _SWAP_RUNTIME_CACHE[key]


def inner_product(left, right, shots=1024, **kwargs):
    r"""
    Estimate the overlap magnitude used in the reduced-gradient assembly.
    """

    # --------------------------------------------------
    # Mode 1: HHL + swap-test readout from adjoint handle
    # --------------------------------------------------
    if isinstance(left, AdjointSwapHandle):
        v_unit, w_norm = _build_test_vector(left, right)

        if w_norm == 0:
            return 0.0

        dim = len(left.rhs_vector)

        hhl, backend, executer, post_processor, state_prep = _get_swap_runtime(
            dim,
            backend_mode=kwargs.get("backend_mode", "aer"),
            ibm_backend_name=kwargs.get("ibm_backend_name"),
            ibm_channel=kwargs.get("ibm_channel"),
            ibm_token=kwargs.get("ibm_token"),
            ibm_instance=kwargs.get("ibm_instance"),
            ibm_use_least_busy=kwargs.get("ibm_use_least_busy", False),
        )

        A = np.asarray(left.system_matrix, dtype=float)
        b = np.asarray(left.rhs_vector, dtype=float)

        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            return 0.0

        b_unit = b / b_norm

        # Build only the core HHL circuit.
        # HHL.build_circuit(A, b, state_prep, *, t0=None, C=None)
        # It does NOT accept swap_test_vector.
        qlsa_circuit = hhl.build_circuit(
            A,
            b_unit,
            state_prep,
        )

        # Attach swap-test readout separately.
        readout = SwapTestReadout(
            swap_test_vector=v_unit,
            state_prep=state_prep,
            post_processor=post_processor,
        )

        circuit = readout.apply(
            qlsa_circuit,
            state_prep=state_prep,
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

        swap_result = readout.process(
            result,
            A,
            b_unit,
            verbose=False,
        )

        if isinstance(swap_result, (tuple, list)):
            exp_value = float(swap_result[0])
        else:
            exp_value = float(swap_result)

        # For the swap test:
        #   P(1) = (1 - |<v|x>|^2) / 2
        # so:
        #   |<v|x>| = sqrt(max(0, 1 - 2 P(1))).
        overlap_mag = np.sqrt(max(0.0, 1.0 - 2.0 * exp_value))

        # Recover the sign classically for testing/debugging.
        sign = _classical_overlap_sign(
            A,
            b_unit,
            v_unit,
        )

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

    backend = _build_backend(
        backend_mode=kwargs.get("backend_mode", "aer"),
        ibm_backend_name=kwargs.get("ibm_backend_name"),
        ibm_channel=kwargs.get("ibm_channel"),
        ibm_token=kwargs.get("ibm_token"),
        ibm_instance=kwargs.get("ibm_instance"),
        ibm_use_least_busy=kwargs.get("ibm_use_least_busy", False),
    )

    result = backend.run(qc, shots=int(shots)).result()
    counts = result.get_counts()

    p0 = counts.get("0", 0) / int(shots)
    overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

    sign = np.sign(np.dot(p_pad, w_pad))
    if sign == 0:
        sign = 1.0

    if left_is_statevector:
        overlap = w_norm * overlap
    else:
        overlap = p_norm * w_norm * overlap

    return float(sign * overlap)