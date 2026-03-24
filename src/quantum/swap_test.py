import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def inner_product(left, right, **kwargs):
    """
    Estimate the overlap |<left|right>| using an exact statevector
    simulation of the swap test.

    Parameters
    ----------
    left : Statevector or np.ndarray
        First state/vector.

    right : np.ndarray
        Second vector/state.

    Returns
    -------
    float
        Exact overlap magnitude |<left|right>|.
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
    # Pad vectors to power-of-two size
    # --------------------------------------------------

    left_pad = np.zeros(state_dim, dtype=complex)
    right_pad = np.zeros(state_dim, dtype=complex)

    left_pad[:len(left_vec)] = left_vec
    right_pad[:len(right_vec)] = right_vec

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
    # Build swap test circuit (no measurement)
    # --------------------------------------------------

    qc = QuantumCircuit(1 + 2 * n_qubits)

    qc.initialize(left_pad, range(1, n_qubits + 1))
    qc.initialize(right_pad, range(n_qubits + 1, 2 * n_qubits + 1))

    qc.h(0)
    for i in range(n_qubits):
        qc.cswap(0, 1 + i, 1 + n_qubits + i)
    qc.h(0)

    # --------------------------------------------------
    # Exact statevector simulation
    # --------------------------------------------------

    final_state = Statevector.from_instruction(qc)

    # --------------------------------------------------
    # Compute P(ancilla = 0)
    # --------------------------------------------------
    #
    # Qiskit basis ordering is little-endian, so qubit 0 is the
    # least-significant bit in the basis index. That means ancilla=0
    # corresponds to even indices, and ancilla=1 to odd indices.
    # --------------------------------------------------

    probs = np.abs(final_state.data) ** 2
    p0 = probs[::2].sum()

    # --------------------------------------------------
    # Recover overlap
    # --------------------------------------------------

    overlap = np.sqrt(max(0.0, 2.0 * p0 - 1.0))

    return overlap