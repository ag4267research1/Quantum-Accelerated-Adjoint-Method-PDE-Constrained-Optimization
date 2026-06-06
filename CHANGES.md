# Changes

## IonQ Backend Support

---

### Changes to QLSAs (the library)

Three files inside `QLSAs/src/qlsas/` were modified to make IonQ work correctly.
IBM and Aer paths are unaffected.

---

#### 1. `QLSAs/src/qlsas/algorithms/hhl/hhl.py` — removed ancilla measurement from core circuit

**What changed:** The line
```python
circ.measure(ancilla_flag_register, ancilla_flag_result)
```
was removed from `HHL._build_core_circuit()`. It is now omitted entirely from the
HHL circuit and is instead added at the very end of `SwapTestReadout.apply()`.

**Why:** The original code measured the success ancilla (`ancilla_flag_register`)
after the eigenvalue-inversion step but **before** inverse QPE. Then
`SwapTestReadout.apply()` appended more quantum gates (the swap test) after that
measurement. This is a **mid-circuit measurement** — a measurement followed by
further quantum operations.

IonQ's backend silently returns all-zero bitstrings for any circuit that contains
a mid-circuit measurement. This caused every single run to produce only `'00': N`
in the counts, which then raised `ValueError: No successful HHL shots` because the
ancilla success bit was always 0.

**Why it is safe to defer:** Inverse QPE is an unconditional uncomputation step. It
is not classically conditioned on the ancilla measurement result — it runs
regardless. So the ancilla can be measured at any point after the eigenvalue
inversion without changing the circuit's logic or output distribution.

---

#### 2. `QLSAs/src/qlsas/readout/swap_test.py` — added both measurements at the end of `apply()`

**What changed:** `SwapTestReadout.apply()` now ends with:
```python
circ.measure(swap_test_ancilla, swap_test_result)
circ.measure(qlsa_circuit.ancilla_register, qlsa_circuit.ancilla_creg)
return circ
```
Both classical registers (`swap_test_result` and `ancilla_flag_result`) are measured
as the last two operations in the circuit — after all quantum gates.

**Why:** Companion change to the `hhl.py` fix above. Since the ancilla measurement
was removed from the HHL circuit, it must be added somewhere. Placing both
measurements here at the very end guarantees there are no mid-circuit measurements
in the final circuit submitted to IonQ.

---

#### 3. `QLSAs/src/qlsas/executer.py` — split `run_qiskit()` into IBM vs non-IBM paths

**What changed:** `Executer.run_qiskit()` previously used
`qiskit_ibm_runtime.SamplerV2` for **all** `BackendV2` backends regardless of
provider. It now branches:

```python
if isinstance(backend, IBMBackend):
    # unchanged — SamplerV2, sessions, error mitigation all preserved
    sampler = Sampler(mode=self._session or backend)
    job = sampler.run([transpiled_circuit], shots=shots)
else:
    from qiskit.primitives import BackendSamplerV2
    job = BackendSamplerV2(backend=backend).run([transpiled_circuit], shots=shots)
```

**Why:** IBM's `qiskit_ibm_runtime.SamplerV2` only understands IBM backends. When
given an IonQ or Aer backend it does not correctly reconstruct named classical
registers in the result, so `join_data(["ancilla_flag_result", "swap_test_result"])`
returned all-zero bitstrings. Qiskit-native `BackendSamplerV2` calls
`backend.run()` directly and correctly reconstructs the named-register
`SamplerPubResult` for any `BackendV2`.

**Note:** Even after this change, IonQ in Mode 1 still bypasses `executer.run()`
entirely in `qlsa_solver.py`. This is because `BackendSamplerV2` internally passes
`memory=True` to `backend.run()`, which IonQ's cloud API does not support and which
causes it to return all-zero counts. The `executer.py` change therefore benefits
Aer and similar local simulators; IonQ uses a direct `backend.run()` call in
`qlsa_solver.py`.

---

### Changes to this project (`src/`)

**`src/quantum/qlsa_solver.py`**

| Change | Reason |
|---|---|
| Removed `from qlsas.post_processor import Post_Processor` | Class removed from qlsas; replaced by free functions |
| `Post_Processor()` removed from `_get_swap_runtime` cache tuple | Same — no longer exists |
| `ClassicalEigOracle` → `UCRYEigOracle` | Renamed in qlsas; `UCRYEigOracle` is the new default |
| `SwapTestReadout(post_processor=…)` arg removed | Constructor signature changed in qlsas |
| Added `ionq_token`, `ionq_backend_name` params throughout | New IonQ backend config |
| Added `"ionq"` and `"ionq_simulator"` cases to `_build_backend()` | IonQ provider support via `qiskit-ionq` |
| Mode 2: transpile circuit before `backend.run()` | IonQ cannot execute `initialize`/`cswap` without basis-gate decomposition |
| Mode 1: IonQ bypasses `executer.run()`, calls `backend.run()` directly | `BackendSamplerV2` passes `memory=True` which IonQ doesn't support |

**`src/experiments/elliptic_experiment.py`** and **`src/experiments/heat_experiment.py`**

Added `ionq_token` and `ionq_backend_name` to `_build_optimize_kwargs()` so the
YAML config fields are forwarded to the quantum solver.

---

### Usage

```yaml
quantum:
  backend_mode: ionq_simulator   # "ionq_simulator" = no credits; "ionq" + ionq_backend_name = QPU
  ionq_token: "your-api-token"
  shots: 4096
```

```bash
pip install qiskit-ionq
```
