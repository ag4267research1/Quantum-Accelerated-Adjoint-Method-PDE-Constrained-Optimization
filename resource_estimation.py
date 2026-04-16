# from __future__ import annotations

# import argparse
# import copy
# import csv
# import math
# import sys
# from pathlib import Path
# from typing import Any

# import matplotlib.pyplot as plt
# import numpy as np
# import yaml
# from matplotlib.ticker import FuncFormatter, LogLocator
# from qiskit.circuit.gate import Gate
# from qiskit_aer import AerSimulator

# # ---------------------------------------------------------------------
# # Repo bootstrap
# # ---------------------------------------------------------------------

# def find_repo_root(start: Path | None = None) -> Path:
#     p = (start or Path.cwd()).resolve()
#     for d in (p, *p.parents):
#         if (d / ".git").exists() or (d / "pyproject.toml").exists() or (d / "src").exists():
#             return d
#     return p


# REPO_ROOT = find_repo_root()
# SRC_DIR = REPO_ROOT / "src"

# if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))


# # ---------------------------------------------------------------------
# # Project imports
# # ---------------------------------------------------------------------

# from qlsas.algorithms.hhl.hhl import HHL
# from qlsas.data_loader import StatePrep
# from qlsas.transpiler import Transpiler

# from src.classical.classical_solver import state_solver
# from src.models.heat_model import HeatModel
# from src.models.elliptic_model import EllipticModel


# # ---------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Resource estimation for the hybrid PDECO quantum circuit."
#     )
#     parser.add_argument(
#         "config",
#         type=str,
#         help="Path to YAML config file.",
#     )
#     return parser.parse_args()


# # ---------------------------------------------------------------------
# # Small helpers
# # ---------------------------------------------------------------------

# def _next_power_of_two(n: int) -> int:
#     return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


# def _pad_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
#     n = len(b)
#     m = _next_power_of_two(n)

#     if m == n:
#         return A, b, n

#     A_pad = np.eye(m, dtype=float)
#     A_pad[:n, :n] = A

#     b_pad = np.zeros(m, dtype=float)
#     b_pad[:n] = b

#     return A_pad, b_pad, n


# def count_two_qubit_gates(circuit) -> int:
#     count = 0
#     for instruction in circuit.data:
#         if isinstance(instruction.operation, Gate) and len(instruction.qubits) == 2:
#             count += 1
#     return count


# def count_active_qubits(circuit) -> int:
#     """
#     Count only the qubits that actually appear in circuit instructions.

#     This avoids reporting the full backend register size after transpilation.
#     """
#     active = set()
#     for instruction in circuit.data:
#         for qubit in instruction.qubits:
#             active.add(qubit)
#     return len(active)


# def format_pow2(x, _pos):
#     if x <= 0:
#         return ""
#     exponent = int(np.log2(x))
#     coeff = x / (2 ** exponent)
#     coeff_str = "" if np.isclose(coeff, 1.0) else f"{coeff:g}"
#     return f"{coeff_str}2^{exponent}"


# def load_config(path: str) -> dict[str, Any]:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# # ---------------------------------------------------------------------
# # Backend resolution from YAML
# # ---------------------------------------------------------------------

# def _build_ibm_service(ibm_cfg: dict[str, Any]):
#     from qiskit_ibm_runtime import QiskitRuntimeService

#     service_kwargs = {}
#     if ibm_cfg.get("channel") is not None:
#         service_kwargs["channel"] = ibm_cfg["channel"]
#     if ibm_cfg.get("token") is not None:
#         service_kwargs["token"] = ibm_cfg["token"]
#     if ibm_cfg.get("instance") is not None:
#         service_kwargs["instance"] = ibm_cfg["instance"]

#     return QiskitRuntimeService(**service_kwargs)


# def resolve_backends(config: dict[str, Any]) -> list[tuple[str, object]]:
#     re_cfg = config.get("resource_estimation", {})
#     backend_specs = re_cfg.get("backends", [])
#     ibm_cfg = re_cfg.get("ibm", {})

#     if not backend_specs:
#         raise ValueError(
#             "No backends were provided under resource_estimation.backends in the YAML."
#         )

#     service = None
#     resolved: list[tuple[str, object]] = []

#     for spec in backend_specs:
#         kind = spec.get("kind", "aer").lower()
#         label = spec.get("label")

#         if kind == "aer":
#             resolved.append((label or "AerSimulator", AerSimulator()))
#             continue

#         if kind == "ibm":
#             if service is None:
#                 service = _build_ibm_service(ibm_cfg)

#             backend_name = spec.get("name")
#             backend_instance = spec.get("instance", ibm_cfg.get("instance"))
#             use_least_busy = bool(spec.get("use_least_busy", ibm_cfg.get("use_least_busy", False)))
#             min_num_qubits = spec.get("min_num_qubits", ibm_cfg.get("min_num_qubits"))

#             if backend_name is not None:
#                 backend = service.backend(backend_name, instance=backend_instance)
#                 resolved.append((label or backend_name, backend))
#                 continue

#             if use_least_busy:
#                 backend = service.least_busy(
#                     operational=True,
#                     simulator=False,
#                     min_num_qubits=min_num_qubits,
#                     instance=backend_instance,
#                 )
#                 resolved.append((label or backend.name, backend))
#                 continue

#             raise ValueError(
#                 "IBM backend entry must provide either 'name' or 'use_least_busy: true'."
#             )

#         else:
#             raise ValueError(f"Unknown backend kind: {kind}")

#     return resolved


# # ---------------------------------------------------------------------
# # PDECO circuit construction
# # ---------------------------------------------------------------------

# def build_model(config: dict[str, Any], size: int):
#     experiment = str(config.get("experiment", "")).lower()

#     if experiment == "heat":
#         model_cfg = config.get("model", {})
#         nx = int(model_cfg.get("nx", 1))
#         model = HeatModel(n=size, nx=nx)
#         x0 = 0.01 * np.ones(model.nx, dtype=float)
#         return model, x0

#     if experiment == "elliptic":
#         cfg_local = copy.deepcopy(config)
#         cfg_local["grid_size"] = size
#         model = EllipticModel(cfg_local)
#         x0 = np.ones(model.num_dofs, dtype=float)
#         return model, x0

#     raise ValueError(
#         "Unsupported experiment type. Expected 'heat' or 'elliptic' in config['experiment']."
#     )


# def build_pdeco_quantum_data(
#     config: dict[str, Any],
#     size: int,
# ) -> dict[str, Any]:
#     """
#     Build the linear-system and swap-test data from the actual hybrid PDECO pipeline.

#     This uses:
#         u = state_solver(model, x0)
#         A = dc/du(u, x0)
#         rhs = dJ/du(u, x0)
#         w_i = dc/dx_i(u, x0)

#     and then prepares the padded, normalized HHL input.
#     """
#     re_cfg = config.get("resource_estimation", {})
#     control_index = int(re_cfg.get("control_index", 0))

#     model, x0 = build_model(config, size)
#     u0 = state_solver(model=model, x=x0)

#     A = np.asarray(model.jacobian(u0, x0), dtype=float)
#     rhs = np.asarray(model.dJ_du(u0, x0), dtype=float).flatten()

#     if rhs.ndim != 1:
#         raise ValueError("dJ_du must return a vector.")

#     rhs_norm = np.linalg.norm(rhs)
#     if rhs_norm == 0:
#         raise ValueError("The adjoint right-hand side has zero norm.")

#     adjoint_matrix = A.T
#     rhs_unit = rhs / rhs_norm

#     A_pad, b_pad, original_dim = _pad_linear_system(adjoint_matrix, rhs_unit)

#     w_i = np.asarray(model.dc_dx_i(u0, x0, control_index), dtype=float).flatten()
#     if w_i.ndim != 1:
#         raise ValueError("dc_dx_i must return a vector.")

#     v = np.zeros(original_dim, dtype=float)
#     v[: len(w_i)] = w_i

#     v_pad = np.zeros(len(b_pad), dtype=float)
#     v_pad[: len(v)] = v

#     v_norm = np.linalg.norm(v_pad)
#     if v_norm == 0:
#         raise ValueError("The chosen dc/dx_i vector has zero norm.")

#     v_unit = v_pad / v_norm

#     return {
#         "model": model,
#         "x0": x0,
#         "u0": u0,
#         "A": A,
#         "rhs": rhs,
#         "adjoint_matrix": adjoint_matrix,
#         "A_pad": A_pad,
#         "b_pad": b_pad,
#         "v_unit": v_unit,
#         "original_dim": original_dim,
#         "padded_dim": len(b_pad),
#     }


# def build_hhl_circuit_from_config(
#     config: dict[str, Any],
#     size: int,
# ):
#     data = build_pdeco_quantum_data(config, size)
#     re_cfg = config.get("resource_estimation", {})

#     readout = str(re_cfg.get("readout", "swap_test"))
#     eig_oracle = str(re_cfg.get("eig_oracle", "classical"))

#     num_qpe_qubits = int(math.log2(data["padded_dim"]))

#     hhl = HHL(
#         state_prep=StatePrep(method="default"),
#         readout=readout,
#         num_qpe_qubits=num_qpe_qubits,
#         eig_oracle=eig_oracle,
#     )

#     if readout == "swap_test":
#         circuit = hhl.build_circuit(
#             data["A_pad"],
#             data["b_pad"],
#             swap_test_vector=data["v_unit"],
#         )
#     else:
#         circuit = hhl.build_circuit(
#             data["A_pad"],
#             data["b_pad"],
#         )

#     return circuit, data


# # ---------------------------------------------------------------------
# # Resource estimation
# # ---------------------------------------------------------------------

# def run_resource_estimation(config: dict[str, Any]) -> dict[str, Any]:
#     re_cfg = config.get("resource_estimation", {})
#     sizes = list(re_cfg.get("sizes", []))
#     if not sizes:
#         raise ValueError("resource_estimation.sizes must be provided in the YAML.")

#     optimization_level = int(re_cfg.get("optimization_level", 3))
#     backends = resolve_backends(config)

#     results = {
#         "sizes": sizes,
#         "backend_data": {label: {"two_qubit_gates": [], "depth": [], "num_qubits": []} for label, _ in backends},
#     }

#     for size in sizes:
#         circuit, data = build_hhl_circuit_from_config(config, size)

#         print(f"\nProblem size = {size}")
#         print(f"  padded dimension = {data['padded_dim']}")
#         print(f"  circuit qubits before transpilation = {circuit.num_qubits}")

#         for label, backend in backends:
#             transpiler = Transpiler(
#                 circuit=circuit,
#                 backend=backend,
#                 optimization_level=optimization_level,
#             )
#             transpiled = transpiler.optimize()

#             two_q = count_two_qubit_gates(transpiled)
#             depth = transpiled.depth()
#             num_qubits = count_active_qubits(transpiled)

#             results["backend_data"][label]["two_qubit_gates"].append(two_q)
#             results["backend_data"][label]["depth"].append(depth)
#             results["backend_data"][label]["num_qubits"].append(num_qubits)

#             print(
#                 f"  backend = {label:<20} "
#                 f"2q gates = {two_q:<8d} "
#                 f"depth = {depth:<8d} "
#                 f"active qubits = {num_qubits}"
#             )

#     return results


# # ---------------------------------------------------------------------
# # Output
# # ---------------------------------------------------------------------

# def save_csv(config: dict[str, Any], results: dict[str, Any]) -> Path:
#     re_cfg = config.get("resource_estimation", {})
#     output_dir = Path(re_cfg.get("output_dir", "resource_estimation_outputs"))
#     output_dir.mkdir(parents=True, exist_ok=True)

#     csv_path = output_dir / re_cfg.get("csv_name", "resource_estimation.csv")

#     backend_labels = list(results["backend_data"].keys())

#     fieldnames = ["problem_size"]
#     for label in backend_labels:
#         fieldnames.extend([
#             f"{label}_two_qubit_gates",
#             f"{label}_depth",
#             f"{label}_num_qubits",
#         ])

#     with csv_path.open("w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for idx, size in enumerate(results["sizes"]):
#             row = {"problem_size": size}
#             for label in backend_labels:
#                 row[f"{label}_two_qubit_gates"] = results["backend_data"][label]["two_qubit_gates"][idx]
#                 row[f"{label}_depth"] = results["backend_data"][label]["depth"][idx]
#                 row[f"{label}_num_qubits"] = results["backend_data"][label]["num_qubits"][idx]
#             writer.writerow(row)

#     return csv_path


# def save_plot(config: dict[str, Any], results: dict[str, Any]) -> Path:
#     re_cfg = config.get("resource_estimation", {})
#     output_dir = Path(re_cfg.get("output_dir", "resource_estimation_outputs"))
#     output_dir.mkdir(parents=True, exist_ok=True)

#     plot_path = output_dir / re_cfg.get("plot_name", "two_qubit_gates_vs_problem_size.png")
#     sizes = results["sizes"]

#     plt.figure(figsize=(8, 5))
#     markers = ["o", "x", "s", "^", "d", "v", "*", "P"]

#     for idx, (label, backend_result) in enumerate(results["backend_data"].items()):
#         marker = markers[idx % len(markers)]
#         plt.plot(
#             sizes,
#             backend_result["two_qubit_gates"],
#             marker=marker,
#             label=label,
#         )

#     gate_budget_lines = re_cfg.get("gate_budget_lines", [5000, 10000])
#     for budget in gate_budget_lines:
#         plt.hlines(
#             budget,
#             sizes[0],
#             sizes[-1],
#             color="grey",
#             linewidth=1.0,
#             linestyle="--",
#             label=f"{budget} gate budget",
#         )

#     ax = plt.gca()
#     ax.set_xscale("log", base=2)
#     ax.xaxis.set_major_locator(LogLocator(base=2))
#     ax.xaxis.set_major_formatter(FuncFormatter(format_pow2))

#     plt.yscale("log")
#     plt.xlabel("Problem size")
#     plt.ylabel("2-qubit gate count")
#     plt.xlim(sizes[0], sizes[-1])
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300)
#     plt.close()

#     return plot_path


# def main():
#     args = parse_args()
#     config = load_config(args.config)

#     results = run_resource_estimation(config)
#     csv_path = save_csv(config, results)
#     plot_path = save_plot(config, results)

#     print(f"\nSaved CSV  to: {csv_path}")
#     print(f"Saved plot to: {plot_path}")


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FuncFormatter, LogLocator
from qiskit.circuit.gate import Gate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists() or (d / "pyproject.toml").exists() or (d / "src").exists():
            return d
    return p


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.transpiler import Transpiler

from src.classical.classical_solver import state_solver
from src.models.heat_model import HeatModel
from src.models.elliptic_model import EllipticModel


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resource estimation for the hybrid PDECO quantum circuit."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def _pad_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(b)
    m = _next_power_of_two(n)

    if m == n:
        return A, b, n

    A_pad = np.eye(m, dtype=float)
    A_pad[:n, :n] = A

    b_pad = np.zeros(m, dtype=float)
    b_pad[:n] = b

    return A_pad, b_pad, n


def count_two_qubit_gates(circuit) -> int:
    count = 0
    for instruction in circuit.data:
        if isinstance(instruction.operation, Gate) and len(instruction.qubits) == 2:
            count += 1
    return count


def count_active_qubits(circuit) -> int:
    """
    Count only the qubits that actually appear in circuit instructions.

    This avoids reporting the full backend register size after transpilation.
    """
    active = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            active.add(qubit)
    return len(active)


def format_pow2(x, _pos):
    if x <= 0:
        return ""
    exponent = int(np.log2(x))
    coeff = x / (2 ** exponent)
    coeff_str = "" if np.isclose(coeff, 1.0) else f"{coeff:g}"
    return f"{coeff_str}2^{exponent}"


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Backend resolution from YAML
# ---------------------------------------------------------------------

def _build_ibm_service(ibm_cfg: dict[str, Any]):
    from qiskit_ibm_runtime import QiskitRuntimeService

    service_kwargs = {}
    if ibm_cfg.get("channel") is not None:
        service_kwargs["channel"] = ibm_cfg["channel"]
    if ibm_cfg.get("token") is not None:
        service_kwargs["token"] = ibm_cfg["token"]
    if ibm_cfg.get("instance") is not None:
        service_kwargs["instance"] = ibm_cfg["instance"]

    return QiskitRuntimeService(**service_kwargs)


def resolve_backends(config: dict[str, Any]) -> list[tuple[str, object]]:
    re_cfg = config.get("resource_estimation", {})
    backend_specs = re_cfg.get("backends", [])
    ibm_cfg = re_cfg.get("ibm", {})

    if not backend_specs:
        raise ValueError(
            "No backends were provided under resource_estimation.backends in the YAML."
        )

    service = None
    resolved: list[tuple[str, object]] = []

    for spec in backend_specs:
        kind = spec.get("kind", "aer").lower()
        label = spec.get("label")

        if kind == "aer":
            resolved.append((label or "AerSimulator", AerSimulator()))
            continue

        if kind == "ibm":
            if service is None:
                service = _build_ibm_service(ibm_cfg)

            backend_name = spec.get("name")
            backend_instance = spec.get("instance", ibm_cfg.get("instance"))
            use_least_busy = bool(spec.get("use_least_busy", ibm_cfg.get("use_least_busy", False)))
            min_num_qubits = spec.get("min_num_qubits", ibm_cfg.get("min_num_qubits"))

            if backend_name is not None:
                backend = service.backend(backend_name, instance=backend_instance)
                resolved.append((label or backend_name, backend))
                continue

            if use_least_busy:
                backend = service.least_busy(
                    operational=True,
                    simulator=False,
                    min_num_qubits=min_num_qubits,
                    instance=backend_instance,
                )
                resolved.append((label or backend.name, backend))
                continue

            raise ValueError(
                "IBM backend entry must provide either 'name' or 'use_least_busy: true'."
            )

        else:
            raise ValueError(f"Unknown backend kind: {kind}")

    return resolved


# ---------------------------------------------------------------------
# PDECO circuit construction
# ---------------------------------------------------------------------

def build_model(config: dict[str, Any], size: int):
    experiment = str(config.get("experiment", "")).lower()

    if experiment == "heat":
        model_cfg = config.get("model", {})
        nx = int(model_cfg.get("nx", 1))
        model = HeatModel(n=size, nx=nx)
        x0 = 0.01 * np.ones(model.nx, dtype=float)
        return model, x0

    if experiment == "elliptic":
        cfg_local = copy.deepcopy(config)
        cfg_local["grid_size"] = size
        model = EllipticModel(cfg_local)
        x0 = np.ones(model.num_dofs, dtype=float)
        return model, x0

    raise ValueError(
        "Unsupported experiment type. Expected 'heat' or 'elliptic' in config['experiment']."
    )


def build_pdeco_quantum_data(
    config: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    """
    Build the linear-system and swap-test data from the actual hybrid PDECO pipeline.

    This uses:
        u = state_solver(model, x0)
        A = dc/du(u, x0)
        rhs = dJ/du(u, x0)
        w_i = dc/dx_i(u, x0)

    and then prepares the padded, normalized HHL input.
    """
    re_cfg = config.get("resource_estimation", {})
    control_index = int(re_cfg.get("control_index", 0))

    model, x0 = build_model(config, size)
    u0 = state_solver(model=model, x=x0)

    A = np.asarray(model.jacobian(u0, x0), dtype=float)
    rhs = np.asarray(model.dJ_du(u0, x0), dtype=float).flatten()

    if rhs.ndim != 1:
        raise ValueError("dJ_du must return a vector.")

    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0:
        raise ValueError("The adjoint right-hand side has zero norm.")

    adjoint_matrix = A.T
    rhs_unit = rhs / rhs_norm

    A_pad, b_pad, original_dim = _pad_linear_system(adjoint_matrix, rhs_unit)

    w_i = np.asarray(model.dc_dx_i(u0, x0, control_index), dtype=float).flatten()
    if w_i.ndim != 1:
        raise ValueError("dc_dx_i must return a vector.")

    v = np.zeros(original_dim, dtype=float)
    v[: len(w_i)] = w_i

    v_pad = np.zeros(len(b_pad), dtype=float)
    v_pad[: len(v)] = v

    v_norm = np.linalg.norm(v_pad)
    if v_norm == 0:
        raise ValueError("The chosen dc/dx_i vector has zero norm.")

    v_unit = v_pad / v_norm

    return {
        "model": model,
        "x0": x0,
        "u0": u0,
        "A": A,
        "rhs": rhs,
        "adjoint_matrix": adjoint_matrix,
        "A_pad": A_pad,
        "b_pad": b_pad,
        "v_unit": v_unit,
        "original_dim": original_dim,
        "padded_dim": len(b_pad),
    }


def build_hhl_circuit_from_config(
    config: dict[str, Any],
    size: int,
):
    data = build_pdeco_quantum_data(config, size)
    re_cfg = config.get("resource_estimation", {})

    readout = str(re_cfg.get("readout", "swap_test"))
    eig_oracle = str(re_cfg.get("eig_oracle", "classical"))

    num_qpe_qubits = int(math.log2(data["padded_dim"]))

    hhl = HHL(
        state_prep=StatePrep(method="default"),
        readout=readout,
        num_qpe_qubits=num_qpe_qubits,
        eig_oracle=eig_oracle,
    )

    if readout == "swap_test":
        circuit = hhl.build_circuit(
            data["A_pad"],
            data["b_pad"],
            swap_test_vector=data["v_unit"],
        )
    else:
        circuit = hhl.build_circuit(
            data["A_pad"],
            data["b_pad"],
        )

    return circuit, data


# ---------------------------------------------------------------------
# Transpilation helper
# ---------------------------------------------------------------------

def transpile_with_fallback(circuit, backend, optimization_level: int):
    """
    Try transpilation at the requested optimization level first.
    If that fails, retry with lower levels.

    This is mainly to avoid rare failures in aggressive unitary synthesis
    when transpiling HHL circuits to IBM backends.
    """
    levels_to_try = []
    for level in [optimization_level, 1, 0]:
        if level not in levels_to_try:
            levels_to_try.append(level)

    last_error = None

    for level in levels_to_try:
        try:
            transpiler = Transpiler(
                circuit=circuit,
                backend=backend,
                optimization_level=level,
            )
            transpiled = transpiler.optimize()
            return transpiled, level
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Transpilation failed for all attempted optimization levels: {levels_to_try}"
    ) from last_error


# ---------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------

def run_resource_estimation(config: dict[str, Any]) -> dict[str, Any]:
    re_cfg = config.get("resource_estimation", {})
    sizes = list(re_cfg.get("sizes", []))
    if not sizes:
        raise ValueError("resource_estimation.sizes must be provided in the YAML.")

    optimization_level = int(re_cfg.get("optimization_level", 3))
    backends = resolve_backends(config)

    results = {
        "sizes": sizes,
        "backend_data": {label: {"two_qubit_gates": [], "depth": [], "num_qubits": []} for label, _ in backends},
    }

    for size in sizes:
        circuit, data = build_hhl_circuit_from_config(config, size)

        print(f"\nProblem size = {size}")
        print(f"  padded dimension = {data['padded_dim']}")
        print(f"  circuit qubits before transpilation = {circuit.num_qubits}")

        for label, backend in backends:
            transpiled, used_level = transpile_with_fallback(
                circuit=circuit,
                backend=backend,
                optimization_level=optimization_level,
            )

            two_q = count_two_qubit_gates(transpiled)
            depth = transpiled.depth()
            num_qubits = count_active_qubits(transpiled)

            results["backend_data"][label]["two_qubit_gates"].append(two_q)
            results["backend_data"][label]["depth"].append(depth)
            results["backend_data"][label]["num_qubits"].append(num_qubits)

            print(
                f"  backend = {label:<20} "
                f"opt_level = {used_level:<2d} "
                f"2q gates = {two_q:<8d} "
                f"depth = {depth:<8d} "
                f"active qubits = {num_qubits}"
            )

    return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_csv(config: dict[str, Any], results: dict[str, Any]) -> Path:
    re_cfg = config.get("resource_estimation", {})
    output_dir = Path(re_cfg.get("output_dir", "resource_estimation_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / re_cfg.get("csv_name", "resource_estimation.csv")

    backend_labels = list(results["backend_data"].keys())

    fieldnames = ["problem_size"]
    for label in backend_labels:
        fieldnames.extend([
            f"{label}_two_qubit_gates",
            f"{label}_depth",
            f"{label}_num_qubits",
        ])

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, size in enumerate(results["sizes"]):
            row = {"problem_size": size}
            for label in backend_labels:
                row[f"{label}_two_qubit_gates"] = results["backend_data"][label]["two_qubit_gates"][idx]
                row[f"{label}_depth"] = results["backend_data"][label]["depth"][idx]
                row[f"{label}_num_qubits"] = results["backend_data"][label]["num_qubits"][idx]
            writer.writerow(row)

    return csv_path


def save_plot(config: dict[str, Any], results: dict[str, Any]) -> Path:
    re_cfg = config.get("resource_estimation", {})
    output_dir = Path(re_cfg.get("output_dir", "resource_estimation_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / re_cfg.get("plot_name", "two_qubit_gates_vs_problem_size.png")
    sizes = results["sizes"]

    plt.figure(figsize=(8, 5))
    markers = ["o", "x", "s", "^", "d", "v", "*", "P"]

    for idx, (label, backend_result) in enumerate(results["backend_data"].items()):
        marker = markers[idx % len(markers)]
        plt.plot(
            sizes,
            backend_result["two_qubit_gates"],
            marker=marker,
            label=label,
        )

    gate_budget_lines = re_cfg.get("gate_budget_lines", [5000, 10000])
    for budget in gate_budget_lines:
        plt.hlines(
            budget,
            sizes[0],
            sizes[-1],
            color="grey",
            linewidth=1.0,
            linestyle="--",
            label=f"{budget} gate budget",
        )

    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.xaxis.set_major_formatter(FuncFormatter(format_pow2))

    plt.yscale("log")
    plt.xlabel("Problem size")
    plt.ylabel("2-qubit gate count")
    plt.xlim(sizes[0], sizes[-1])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def main():
    args = parse_args()
    config = load_config(args.config)

    results = run_resource_estimation(config)
    csv_path = save_csv(config, results)
    plot_path = save_plot(config, results)

    print(f"\nSaved CSV  to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()