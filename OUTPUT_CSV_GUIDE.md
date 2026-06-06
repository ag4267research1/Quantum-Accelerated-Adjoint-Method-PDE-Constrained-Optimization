# Output CSV Guide

Two CSV files are generated per hybrid experiment run, saved in the experiment output directory.

---

## 1. `hybrid_quantum_calls.csv`

One row per quantum subroutine call. Since there are $n_x$ control variables, each optimizer iteration produces exactly $n_x$ rows in this file.

### Columns

| Column | Description |
|---|---|
| `n` | Number of PDE state variables (nu) |
| `nx` | Number of control variables |
| `iteration` | Optimizer iteration index $k$ |
| `call_within_iteration` | Index of this call within iteration $k$, from $0$ to $n_x - 1$ |
| `shots` | Total number of times the combined HHL + swap test circuit was executed, $S$ |
| `successful_shots` | Number of shots where the HHL ancilla was measured as $\|1\rangle$, $S_{\text{succ}}$ |
| `success_rate` | Fraction of successful post-selections |
| `residual` | Deviation of quantum result from classical ground truth |

### Formulas

**Success rate:**

$$\text{success\_rate} = \frac{S_{\text{succ}}}{S}$$

where $S_{\text{succ}}$ is the number of shots with HHL ancilla $= |1\rangle$ and $S$ is the total shots submitted.

**Residual:**

$$\text{residual} = \left| P(\text{swap}=1)_{\text{measured}} - P(\text{swap}=1)_{\text{classical}} \right|$$

where:

$$P(\text{swap}=1)_{\text{measured}} = \frac{\text{num\_swap\_ones}}{S_{\text{succ}}}$$

$$P(\text{swap}=1)_{\text{classical}} = \frac{1}{2} - \frac{1}{2} \left| \langle v | x \rangle \right|^2$$

Here $|x\rangle = A^{-1}b / \|A^{-1}b\|$ is the normalised classical solution and $|v\rangle$ is the normalised swap test vector. The classical solution is computed internally via `numpy.linalg.solve(A, b)`.

---

## 2. `hybrid_superimposed_iteration_history.csv`

One row per optimizer iteration per state size $n$. The quantum metrics are aggregated over the $n_x$ calls that occur within each iteration.

### Columns

| Column | Description |
|---|---|
| `mode` | Solver mode (`hybrid` or `classical`) |
| `n` | Number of PDE state variables (nu) |
| `nx` | Number of control variables |
| `iteration` | Optimizer iteration index $k$ |
| `objective` | Objective function value $J(u, x)$ at iteration $k$ |
| `gradient_norm` | Euclidean norm of the reduced gradient $\|\nabla J\|$ |
| `condition_number` | Condition number of the Jacobian matrix $\kappa(A) = \|A\| \cdot \|A^{-1}\|$ |
| `step_size` | Actual step length $\alpha_k$ accepted by the Armijo line search |
| `shots_per_iteration` | Total shots across all $n_x$ calls: $S \times n_x$ |
| `successful_shots` | Total successful HHL shots across all $n_x$ calls: $\sum_{i=1}^{n_x} S_{\text{succ},i}$ |
| `avg_success_rate` | Mean success rate across the $n_x$ calls |
| `avg_residual` | Mean swap test residual across the $n_x$ calls |
| `backtrack_count` | Number of times the step was halved before Armijo condition was satisfied |
| `accepted_step` | $1$ if a valid step was found, $0$ if backtracking failed |

### Formulas

**Objective:** $J(u, x)$ is the PDE-constrained cost functional evaluated at state $u$ and control $x$.

**Gradient norm:**

$$\|\nabla J\|_2 = \left\| \frac{dJ}{dx} - p^T \frac{dc}{dx} \right\|_2$$

where $p$ is the adjoint variable solving $C_u^T p = J_u^T$.

**Armijo step size:** The step $\alpha_k$ is the largest value in $\{\alpha \cdot \tau^j : j = 0, 1, 2, \ldots\}$ satisfying:

$$J(x_k - \alpha_k \nabla J_k) \leq J(x_k) - c_1 \alpha_k \|\nabla J_k\|^2$$

where $\tau = 0.5$ is the backtracking factor and $c_1$ is the Armijo constant.

**Average success rate per iteration:**

$$\text{avg\_success\_rate}_k = \frac{1}{n_x} \sum_{i=1}^{n_x} \frac{S_{\text{succ},i}}{S}$$

**Average residual per iteration:**

$$\text{avg\_residual}_k = \frac{1}{n_x} \sum_{i=1}^{n_x} \left| P(\text{swap}=1)_{\text{measured},i} - P(\text{swap}=1)_{\text{classical},i} \right|$$

---

## Relationship Between the Two Files

Each row in `hybrid_superimposed_iteration_history.csv` at iteration $k$ corresponds to exactly $n_x$ rows in `hybrid_quantum_calls.csv` where `iteration = k` and `call_within_iteration` $\in \{0, 1, \ldots, n_x - 1\}$.

For classical mode, all quantum columns (`successful_shots`, `avg_success_rate`, `avg_residual`) are `NaN`.
