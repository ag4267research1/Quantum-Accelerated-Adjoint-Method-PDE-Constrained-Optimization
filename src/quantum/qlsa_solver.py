########### Statevector ~ HHL version ##############

# import numpy as np

# from qlsas.algorithms.hhl.hhl import HHL
# from qlsas.data_loader import StatePrep
# from qiskit.quantum_info import Statevector


# # ----------------------------------------------------------
# # Global caches
# # ----------------------------------------------------------

# _HHL_CACHE = {}
# _CIRCUIT_CACHE = {}


# def _next_power_of_two(n):
#     """
#     Return smallest power of two >= n.
#     Quantum state vectors must have size 2^m.
#     """
#     return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


# def _pad_linear_system(A, b):
#     """
#     Pad Ax=b so the dimension is a power of two.
#     """

#     n = len(b)
#     m = _next_power_of_two(n)

#     if m == n:
#         return A, b, n

#     A_pad = np.eye(m, dtype=float)
#     A_pad[:n, :n] = A

#     b_pad = np.zeros(m, dtype=float)
#     b_pad[:n] = b

#     return A_pad, b_pad, n


# def _get_hhl_instance(dim):
#     """
#     Return cached HHL instance for a given dimension.
#     """

#     if dim not in _HHL_CACHE:

#         hhl = HHL(
#             state_prep=StatePrep(method="default"),
#             readout="measure_x",
#             num_qpe_qubits=int(np.log2(dim)),
#             eig_oracle="classical"
#         )

#         _HHL_CACHE[dim] = hhl

#     return _HHL_CACHE[dim]


# def _remove_measurements(circuit):
#     """
#     Remove all measurement instructions so the circuit
#     can be simulated as a statevector.
#     """

#     cleaned = circuit.copy()

#     cleaned.data = [
#         inst for inst in cleaned.data
#         if inst.operation.name != "measure"
#     ]

#     return cleaned


# def _get_hhl_circuit(hhl, A, b):
#     """
#     Cache the HHL circuit so we don't rebuild it every
#     iteration of the optimizer.
#     """

#     key = (A.shape[0],)

#     if key not in _CIRCUIT_CACHE:

#         circuit = hhl.build_circuit(A, b)

#         # remove measurements for statevector simulation
#         circuit = _remove_measurements(circuit)

#         _CIRCUIT_CACHE[key] = circuit

#     return _CIRCUIT_CACHE[key]


# def adjoint_solver(A, rhs, **kwargs):
#     """
#     Solve the adjoint system

#         A^T p = rhs

#     Returns the quantum statevector |p> suitable for
#     use in a swap test.
#     """

#     # --------------------------------------------------
#     # Form adjoint system
#     # --------------------------------------------------

#     AT = A.T

#     # --------------------------------------------------
#     # Normalize RHS
#     # --------------------------------------------------

#     rhs_norm = np.linalg.norm(rhs)

#     if rhs_norm == 0.0:
#         return np.zeros_like(rhs)

#     b = rhs / rhs_norm

#     # --------------------------------------------------
#     # Pad system
#     # --------------------------------------------------

#     AT_pad, b_pad, original_dim = _pad_linear_system(AT, b)

#     dim = len(b_pad)

#     # --------------------------------------------------
#     # Reuse HHL instance
#     # --------------------------------------------------

#     hhl = _get_hhl_instance(dim)

#     # --------------------------------------------------
#     # Reuse cached circuit
#     # --------------------------------------------------

#     circuit = _get_hhl_circuit(hhl, AT_pad, b_pad)

#     # --------------------------------------------------
#     # Simulate quantum state
#     # --------------------------------------------------

#     state = Statevector.from_instruction(circuit)

#     vec = state.data

#     # Extract solution amplitudes
#     p_state = vec[:dim]

#     # Remove padding
#     p_state = p_state[:original_dim]

#     # Undo normalization
#     p_state = p_state * rhs_norm

#     return p_state


import numpy as np

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.solver import QuantumLinearSolver
from qiskit_aer import AerSimulator


# ----------------------------------------------------------
# Global caches
# ----------------------------------------------------------

# _HHL_CACHE = {}
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
            eig_oracle="classical"
        )

        backend = AerSimulator()

        solver = QuantumLinearSolver(
            qlsa=hhl,
            backend=backend,
            shots=shots,
            optimization_level=0
        )

        _SOLVER_CACHE[key] = solver

    return _SOLVER_CACHE[key]


def adjoint_solver(A, rhs, shots=1024, **kwargs):
    """
    Solve the adjoint system

        A^T p = rhs
    """

    # --------------------------------------------------
    # Form adjoint system
    # --------------------------------------------------

    AT = A.T

    # --------------------------------------------------
    # Normalize RHS
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

    solver = _get_solver(dim, shots)

    # --------------------------------------------------
    # Solve using HHL framework
    # --------------------------------------------------

    solution = solver.solve(AT_pad, b_pad)

    # --------------------------------------------------
    # Remove padding
    # --------------------------------------------------

    p = solution[:original_dim]

    # --------------------------------------------------
    # Undo RHS normalization
    # --------------------------------------------------

    # p = p * rhs_norm

    return p