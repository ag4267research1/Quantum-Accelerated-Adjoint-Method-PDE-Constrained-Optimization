"""
Quantum adjoint solver for PDE-constrained optimization.

Public API
----------
adjoint_solver(A, rhs, ...) -> (AdjointSwapHandle, p_scale)
    Prepares the padded adjoint system for HHL. The norm ||p|| is computed
    classically; the handle carries the system data for later inner product calls.

inner_product(left, right, shots, ...) -> float
    Mode 1 (AdjointSwapHandle): estimates <p, w_i> via HHL + swap test.
      Overlap magnitude comes from the quantum circuit; sign is recovered classically.
    Mode 2 (Statevector or ndarray): runs a standalone swap-test circuit directly.
"""

import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.state_prep import DefaultStatePrep
from qlsas.executer import Executer
from qlsas.transpiler import Transpiler
from qlsas.algorithms.hhl.eig_oracles import UCRYEigOracle
from qlsas.readout.swap_test import SwapTestReadout


# ----------------------------------------------------------
# Data structure passed from adjoint_solver to inner_product
# ----------------------------------------------------------

@dataclass
class AdjointSwapHandle:
    """
    Lightweight container describing the HHL problem instance for a later
    swap-test overlap query.
    """
    system_matrix: np.ndarray   # padded adjoint matrix passed to HHL
    rhs_vector: np.ndarray      # normalized padded RHS passed to HHL
    original_dim: int           # unpadded dimension of the system
    state_dim: int              # original adjoint dimension n
    shots: int                  # default shot count for the swap test
    p_scale: float              # classical norm ||p|| of the adjoint vector


# ----------------------------------------------------------
# Backend construction and runtime cache
# ----------------------------------------------------------

_SWAP_RUNTIME_CACHE = {}


def _is_ionq_backend(backend):
    return type(backend).__module__.startswith("qiskit_ionq")


def _build_backend(
    backend_mode="aer",
    ibm_backend_name=None,
    ibm_channel=None,
    ibm_token=None,
    ibm_instance=None,
    ibm_use_least_busy=False,
    ionq_token=None,
    ionq_backend_name=None,
):
    """Build a local Aer simulator, an IBM Quantum backend, or an IonQ backend."""
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

    if backend_mode in ("ionq", "ionq_simulator"):
        try:
            from qiskit_ionq import IonQProvider
        except ImportError as exc:
            raise ImportError(
                "qiskit-ionq is required for backend_mode='ionq'/'ionq_simulator'. "
                "Install it with: pip install qiskit-ionq"
            ) from exc

        if ionq_token is None:
            raise ValueError(
                f"ionq_token is required for backend_mode='{backend_mode}'."
            )

        provider = IonQProvider(token=ionq_token)
        if backend_mode == "ionq_simulator":
            name = "ionq_simulator"
        else:
            name = ionq_backend_name if ionq_backend_name is not None else "ionq_simulator"
        return provider.get_backend(name)

    raise ValueError(f"Unknown backend_mode: {backend_mode}")


def _get_swap_runtime(
    dim,
    backend_mode="aer",
    ibm_backend_name=None,
    ibm_channel=None,
    ibm_token=None,
    ibm_instance=None,
    ibm_use_least_busy=False,
    ionq_token=None,
    ionq_backend_name=None,
):
    """Return cached (hhl, backend, executer, post_processor, state_prep) for dim."""
    key = (
        int(dim),
        backend_mode,
        ibm_backend_name,
        ibm_channel,
        ibm_instance,
        bool(ibm_use_least_busy),
        ionq_backend_name,
    )

    if key not in _SWAP_RUNTIME_CACHE:
        hhl = HHL(
            num_qpe_qubits=int(np.log2(dim)),
            eig_oracle=UCRYEigOracle(),
        )

        backend = _build_backend(
            backend_mode=backend_mode,
            ibm_backend_name=ibm_backend_name,
            ibm_channel=ibm_channel,
            ibm_token=ibm_token,
            ibm_instance=ibm_instance,
            ibm_use_least_busy=ibm_use_least_busy,
            ionq_token=ionq_token,
            ionq_backend_name=ionq_backend_name,
        )

        _SWAP_RUNTIME_CACHE[key] = (
            hhl,
            backend,
            Executer(),
            DefaultStatePrep(),
        )

    return _SWAP_RUNTIME_CACHE[key]


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def _next_power_of_two(n):
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A, b):
    r"""
    Pad a square linear system to the next power-of-two dimension:

        A_pad = [[A, 0], [0, I]],   b_pad = [b, 0]

    The identity block prevents singular padding directions.
    Returns (A_pad, b_pad, original_n).
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
    """Pad to power-of-two, normalize, and return (Statevector, original_norm)."""
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
    """Solve C_u^T p = J_u^T classically and return ||p||."""
    try:
        p = np.linalg.solve(C_u_T, J_u_T)
    except np.linalg.LinAlgError:
        p = np.linalg.lstsq(C_u_T, J_u_T, rcond=None)[0]
    return float(np.linalg.norm(p))


def _classical_overlap_sign(system_matrix, rhs_vector, swap_test_vector):
    """
    Recover the sign of <swap_test_vector, x> via a classical solve.
    HHL returns |<v|x>|; sign must be resolved separately.
    """
    try:
        x = np.linalg.solve(system_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(system_matrix, rhs_vector, rcond=None)[0]

    sign = np.sign(np.dot(swap_test_vector, x))
    return float(sign) if sign != 0 else 1.0


def _build_test_vector(handle, right):
    """
    Build the normalized swap-test vector in the padded HHL space.
    Returns (v_unit, w_norm) where w_norm is the pre-normalization scale.
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

    return v_pad / w_norm, w_norm


# ----------------------------------------------------------
# Public API 1: adjoint solve preparation
# ----------------------------------------------------------

def adjoint_solver(
    A,
    rhs,
    shots=2048,
    use_preconditioning=False,  # kept for interface compatibility; not used
    eps=1e-12,
    check_hermitian=True,
    return_diagnostics=False,
    **kwargs,
):
    r"""
    Prepare the adjoint system  C_u^T p = J_u^T  for HHL + swap-test queries.

    Returns (AdjointSwapHandle, p_scale) where p_scale = ||p|| is solved
    classically and the handle carries the padded system for inner_product calls.
    """
    C_u_T = np.asarray(A, dtype=float).T
    J_u_T = np.asarray(rhs, dtype=float).copy()

    cond_raw = np.linalg.cond(C_u_T)
    p_scale = _classical_adjoint_norm(C_u_T, J_u_T)

    rhs_norm = np.linalg.norm(J_u_T)
    if rhs_norm == 0:
        p_state, p_scale_zero = _vector_to_state_and_norm(np.zeros_like(J_u_T))
        if return_diagnostics:
            return p_state, p_scale_zero, {
                "cond_raw": cond_raw, "cond_pad": cond_raw,
                "rhs_norm": 0.0, "used_preconditioning": False,
            }
        return p_state, p_scale_zero

    if check_hermitian and not np.allclose(C_u_T, C_u_T.T, atol=1e-10):
        raise ValueError("Adjoint matrix passed to HHL is not symmetric.")

    b_unit = J_u_T / rhs_norm
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
            "cond_raw": cond_raw, "cond_pad": cond_pad,
            "rhs_norm": rhs_norm, "used_preconditioning": False,
        }

    return handle, p_scale


# ----------------------------------------------------------
# Public API 2: overlap estimation
# ----------------------------------------------------------

def inner_product(left, right, shots=1024, **kwargs):
    r"""
    Estimate the inner product used in reduced-gradient assembly.

    Mode 1 (AdjointSwapHandle): runs HHL + swap test to estimate |<v|x>|,
    then recovers the sign classically.

    Mode 2 (Statevector or ndarray): runs a standalone swap-test circuit
    directly on the two vectors.
    """

    # --------------------------------------------------
    # Mode 1: HHL + swap-test from adjoint handle
    # --------------------------------------------------
    if isinstance(left, AdjointSwapHandle):
        v_unit, w_norm = _build_test_vector(left, right)
        if w_norm == 0:
            return 0.0

        dim = len(left.rhs_vector)
        hhl, backend, executer, state_prep = _get_swap_runtime(
            dim,
            backend_mode=kwargs.get("backend_mode", "aer"),
            ibm_backend_name=kwargs.get("ibm_backend_name"),
            ibm_channel=kwargs.get("ibm_channel"),
            ibm_token=kwargs.get("ibm_token"),
            ibm_instance=kwargs.get("ibm_instance"),
            ibm_use_least_busy=kwargs.get("ibm_use_least_busy", False),
            ionq_token=kwargs.get("ionq_token"),
            ionq_backend_name=kwargs.get("ionq_backend_name"),
        )

        A = np.asarray(left.system_matrix, dtype=float)
        b = np.asarray(left.rhs_vector, dtype=float)

        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            return 0.0
        b_unit = b / b_norm

        qlsa_circuit = hhl.build_circuit(A, b_unit, state_prep)

        readout = SwapTestReadout(
            swap_test_vector=v_unit,
            state_prep=state_prep,
        )
        circuit = readout.apply(qlsa_circuit, state_prep=state_prep)

        if _is_ionq_backend(backend):
            # IonQ doesn't preserve named classical registers in results, so
            # join_data() inside readout.process() returns all-zero bitstrings.
            # Use AerSimulator for HHL+swap-test execution which correctly
            # handles named registers and mid-circuit measurements.
            _exec_backend = AerSimulator()
        else:
            _exec_backend = backend

        transpiled_circuit = Transpiler(
            circuit=circuit, backend=_exec_backend, optimization_level=0
        ).optimize()
        result = executer.run(transpiled_circuit, _exec_backend, int(shots), verbose=False)
        swap_result = readout.process(result, A, b_unit, verbose=False)

        exp_value = float(swap_result[0] if isinstance(swap_result, (tuple, list)) else swap_result)

        # Swap test:  P(1) = (1 - |<v|x>|²) / 2  →  |<v|x>| = sqrt(1 - 2·P(1))
        overlap_mag = np.sqrt(max(0.0, 1.0 - 2.0 * exp_value))
        sign = _classical_overlap_sign(A, b_unit, v_unit)
        return float(sign * overlap_mag * w_norm)

    # --------------------------------------------------
    # Mode 2: standalone swap test on two vectors
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
        ionq_token=kwargs.get("ionq_token"),
        ionq_backend_name=kwargs.get("ionq_backend_name"),
    )

    result = backend.run(qc, shots=int(shots)).result()
    counts = result.get_counts()

    p0 = counts.get("0", 0) / int(shots)
    overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

    sign = np.sign(np.dot(p_pad, w_pad))
    if sign == 0:
        sign = 1.0

    if left_is_statevector:
        return float(sign * w_norm * overlap)
    return float(sign * p_norm * w_norm * overlap)
