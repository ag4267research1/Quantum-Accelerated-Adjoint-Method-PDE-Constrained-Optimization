import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def inner_product(left, right, shots=1024, **kwargs):
    """
    Swap test between prepared quantum state |p>
    and classical vector w_i.
    """

    # --------------------------------------------------
    # Extract vector from quantum state if necessary
    # --------------------------------------------------

    if isinstance(left, Statevector):
        p_vec = left.data
        left_is_statevector = True   # CHANGED: track whether left is already a normalized quantum state
    else:
        p_vec = np.asarray(left, dtype=complex)
        left_is_statevector = False  # CHANGED

    w_vec = np.asarray(right, dtype=complex)

    # --------------------------------------------------
    # Determine number of qubits
    # --------------------------------------------------

    n = int(np.ceil(np.log2(max(len(p_vec), len(w_vec)))))
    size = 2 ** n

    # --------------------------------------------------
    # Pad vectors
    # --------------------------------------------------

    p_pad = np.zeros(size, dtype=complex)
    w_pad = np.zeros(size, dtype=complex)

    # p_pad[:len(p_vec)] = np.real(p_vec)
    p_pad[:len(p_vec)] = p_vec
    w_pad[:len(w_vec)] = w_vec

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    p_norm = np.linalg.norm(p_pad)
    w_norm = np.linalg.norm(w_pad)

    # if p_norm > 0:
    #     p_pad = p_pad / p_norm

    # if w_norm > 0:
    #     w_pad = w_pad / w_norm
    
    if p_norm == 0 or w_norm == 0:
        return 0.0

    p_pad = p_pad / p_norm
    w_pad = w_pad / w_norm

    # --------------------------------------------------
    # Build swap test circuit
    # --------------------------------------------------

    qc = QuantumCircuit(1 + 2 * n, 1)

    qc.initialize(p_pad, range(1, n + 1))
    qc.initialize(w_pad, range(n + 1, 2 * n + 1))
    
    # if isinstance(left, Statevector):
    #     p_vec = left.data
    # elif isinstance(left, np.ndarray):
    #     p_vec = left
    # else:
    #     raise ValueError("left must be a statevector or vector")

    qc.h(0)

    for i in range(n):
        qc.cswap(0, 1 + i, 1 + n + i)

    qc.h(0)

    qc.measure(0, 0)

    # --------------------------------------------------
    # Execute
    # --------------------------------------------------

    backend = AerSimulator()

    result = backend.run(qc, shots=shots).result()

    counts = result.get_counts()

    p0 = counts.get("0", 0) / shots

    # --------------------------------------------------
    # Recover overlap
    # --------------------------------------------------

    overlap = np.sqrt(max(0.0, 2 * p0 - 1))

    # CHANGED: if left is a Statevector from qlsa_solver, it is already the
    # normalized adjoint state |p>. The optimizer multiplies by p_scale outside,
    # so here we should only restore the norm of w_i.
    # If left is a classical vector, we restore both norms.
    if left_is_statevector:
        overlap = w_norm * overlap
    else:
        overlap = p_norm * w_norm * overlap

    # CHANGED: the plain swap test only gives the magnitude of the overlap.
    # The optimizer needs a signed scalar, so we recover the sign from the
    # real part of the padded-vector inner product.
    sign = np.sign(np.real(np.vdot(p_pad, w_pad)))
    if sign == 0:
        sign = 1.0

    overlap = float(sign * overlap)

    return overlap