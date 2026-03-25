import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService


# ----------------------------------------------------------
# Global cache
# ----------------------------------------------------------
_SERVICE_CACHE = None
_BACKEND_CACHE = {}


def _get_runtime_service():
    """
    Return a cached IBM Runtime service object.

    This assumes you have already saved your IBM Quantum Platform
    credentials using QiskitRuntimeService.save_account(...).
    """
    global _SERVICE_CACHE

    if _SERVICE_CACHE is None:
        _SERVICE_CACHE = QiskitRuntimeService()

    return _SERVICE_CACHE


def _get_backend(backend_mode="aer", backend_name=None):
    """
    Return either a local Aer simulator backend or an IBM backend.

    Parameters
    ----------
    backend_mode : str
        "aer" or "ibm"

    backend_name : str or None
        Optional backend name. If backend_mode="ibm" and backend_name
        is None, the least busy operational backend is chosen.
    """
    key = (backend_mode, backend_name)

    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    if backend_mode == "aer":
        backend = AerSimulator()

    elif backend_mode == "ibm":
        service = _get_runtime_service()

        if backend_name is not None:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)

    else:
        raise ValueError(f"Unknown backend_mode: {backend_mode}")

    _BACKEND_CACHE[key] = backend
    return backend


def inner_product(
    left,
    right,
    shots=1024,
    backend_mode="aer",
    backend_name=None,
    **kwargs,
):
    """
    Estimate the overlap |<left|right>| using a swap test.

    Parameters
    ----------
    left : Statevector or np.ndarray
        First state/vector. If a NumPy array is provided, it is treated
        as a state-amplitude vector.

    right : np.ndarray
        Second vector/state.

    shots : int
        Number of shots used for measurement-based estimation.

    backend_mode : str
        "aer" for local simulation, "ibm" for IBM backend execution.

    backend_name : str or None
        Specific IBM backend name if backend_mode="ibm".

    Returns
    -------
    float
        Estimated overlap magnitude |<left|right>|.
    """

    # --------------------------------------------------
    # Convert inputs to complex vectors
    # --------------------------------------------------

    if isinstance(left, Statevector):
        left_vec = left.data
    else:
        left_vec = np.asarray(left, dtype=complex)

    right_vec = np.asarray(right, dtype=complex)

    # --------------------------------------------------
    # Determine number of qubits
    # --------------------------------------------------

    max_len = max(len(left_vec), len(right_vec))
    n_qubits = int(np.ceil(np.log2(max_len)))
    state_dim = 2 ** n_qubits

    # --------------------------------------------------
    # Pad vectors to power-of-two dimension
    # --------------------------------------------------

    left_pad = np.zeros(state_dim, dtype=complex)
    right_pad = np.zeros(state_dim, dtype=complex)

    left_pad[: len(left_vec)] = left_vec
    right_pad[: len(right_vec)] = right_vec

    # --------------------------------------------------
    # Normalize states
    # --------------------------------------------------

    left_norm = np.linalg.norm(left_pad)
    right_norm = np.linalg.norm(right_pad)

    if left_norm == 0 or right_norm == 0:
        return 0.0

    left_pad /= left_norm
    right_pad /= right_norm

    # --------------------------------------------------
    # Build swap test circuit
    # --------------------------------------------------

    qc = QuantumCircuit(1 + 2 * n_qubits, 1)

    # Register layout:
    # qubit 0                  -> ancilla
    # qubits 1 ... n_qubits    -> |left>
    # qubits n_qubits+1 ...    -> |right>

    qc.initialize(left_pad, range(1, n_qubits + 1))
    qc.initialize(right_pad, range(n_qubits + 1, 2 * n_qubits + 1))

    qc.h(0)

    for i in range(n_qubits):
        qc.cswap(0, 1 + i, 1 + n_qubits + i)

    qc.h(0)
    qc.measure(0, 0)

    # --------------------------------------------------
    # Execute
    # --------------------------------------------------

    backend = _get_backend(
        backend_mode=backend_mode,
        backend_name=backend_name,
    )

    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Probability of measuring ancilla = 0
    p0 = counts.get("0", 0) / shots

    # --------------------------------------------------
    # Recover overlap
    # --------------------------------------------------
    #
    # Swap test formula:
    #   P(0) = (1 + |<left|right>|^2) / 2
    #
    # Therefore:
    #   |<left|right>| = sqrt(2 P(0) - 1)
    # --------------------------------------------------

    overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

    return overlap