# Changes

## IonQ Backend Support

### Problem

`qlsas/executer.py` used `qiskit_ibm_runtime.SamplerV2` for **all** `BackendV2`
backends. IBM's runtime sampler does not return named-register data for non-IBM
backends (IonQ, etc.), so `join_data(["ancilla_flag_result", "swap_test_result"])`
inside `readout.process()` produced all-zero bitstrings, causing every run to
raise `ValueError: No successful HHL shots`.

### Root cause

The HHL circuit measured the success ancilla (`ancilla_flag_result`) **before** the
inverse QPE uncomputation step, then `SwapTestReadout` appended the swap-test gates
after that. This created a mid-circuit measurement — quantum gates continued after
the ancilla was measured. IonQ's qiskit provider silently returns all-zero results
for circuits containing mid-circuit measurements, causing every run to raise
`ValueError: No successful HHL shots`.

Because the inverse QPE is unconditional (not classically conditioned on the ancilla
result), the measurement is purely for post-selection and can safely be deferred.

### Fix

**`QLSAs/src/qlsas/algorithms/hhl/hhl.py`** — `HHL.build_circuit()` (2 lines swapped)

Swapped steps 4 and 5: inverse QPE now runs before the ancilla measurement, making
the ancilla measurement a terminal measurement. No logic change — inverse QPE was
never conditioned on the ancilla result.

**`QLSAs/src/qlsas/executer.py`** — `Executer.run_qiskit()` (~5 lines)

Branch on `IBMBackend`:
- IBM backends → keep existing `qiskit_ibm_runtime.SamplerV2` path (sessions,
  error mitigation options, etc. unchanged)
- All other backends (Aer, IonQ, …) → use Qiskit-native `BackendSamplerV2`,
  which calls `backend.run()` directly and correctly reconstructs named-register
  `SamplerPubResult` from any `BackendV2`

**`src/quantum/qlsa_solver.py`** — several changes across the session

| Change | Reason |
|---|---|
| Removed `from qlsas.post_processor import Post_Processor` | Class removed from qlsas; replaced by free functions |
| `Post_Processor()` removed from `_get_swap_runtime` cache tuple | Same — no longer exists |
| `ClassicalEigOracle` → `UCRYEigOracle` | Renamed in qlsas; `UCRYEigOracle` is the new default |
| `SwapTestReadout(post_processor=…)` arg removed | Constructor signature changed in qlsas |
| Added `ionq_token`, `ionq_backend_name` params throughout | New IonQ backend config |
| Added `"ionq"` and `"ionq_simulator"` cases to `_build_backend()` | IonQ provider support via `qiskit-ionq` |
| Removed Aer substitution workaround for IonQ | No longer needed after qlsas fix |
| Mode 2 (`inner_product`): transpile circuit before `backend.run()` | IonQ cannot execute `initialize`/`cswap` without basis-gate decomposition |
| Mode 1 (`inner_product`): IonQ bypasses `executer.run()`, calls `backend.run()` directly | `BackendSamplerV2` internally passes `memory=True` which IonQ doesn't support, returning all-zero counts |

**`src/experiments/elliptic_experiment.py`** and **`src/experiments/heat_experiment.py`**

Added `ionq_token` and `ionq_backend_name` to `_build_optimize_kwargs()` so the
YAML config fields are forwarded to the quantum solver.

### Usage

```yaml
quantum:
  backend_mode: ionq_simulator   # or "ionq" + ionq_backend_name for QPU
  ionq_token: "your-api-token"
  shots: 4096
```

Install the IonQ provider:

```bash
pip install qiskit-ionq
```
