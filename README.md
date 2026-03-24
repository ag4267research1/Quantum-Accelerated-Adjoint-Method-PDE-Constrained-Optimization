# hybrid-PDECO


This project implements a hybrid quantum--classical framework for solving 
PDE-constrained optimization problems using Quantum Linear System Algorithms (QLSAs). 
In particular, we use an existing QLSA software implementation that provides 
quantum circuit constructions and execution pipelines for solving linear systems 
using the HHL (Harrow--Hassidim--Lloyd) algorithm.

In many PDE-constrained optimization problems, computing gradients requires solving 
large linear systems of the form

$$
A^T p = \frac{\partial J}{\partial u},
$$

where $A$ is the Jacobian of the PDE residual with respect to the state variables 
and $p$ is the adjoint variable. Classical approaches solve these systems using 
numerical linear algebra methods whose computational cost scales polynomially with 
the system size.

In this work, we investigate whether quantum linear system algorithms can accelerate 
this step. The implementation integrates a classical PDE solver with a QLSA-based 
linear system solver by embedding the HHL algorithm within the adjoint computation 
of the optimization loop.

The computational workflow follows the hybrid pipeline

$$
\text{PDE Residual Solver}
\rightarrow
\text{Jacobian Construction}
\rightarrow
\text{Adjoint Linear System}
\rightarrow
\text{Quantum Linear Solver (HHL)}
\rightarrow
\text{Gradient Computation}
\rightarrow
\text{Optimization Update}.
$$

Experiments in this project focus on validating adjoint gradients computed using 
HHL, comparing classical and quantum adjoint solutions, and studying the scaling 
behavior of the hybrid solver.

Overall, this project explores the integration of quantum linear system algorithms 
with PDE-constrained optimization as a step toward understanding the potential of 
quantum acceleration in scientific computing.


## Project Structure
<!-- tree -I "__pycache__|*.pyc" -->

```
project-root
├── README.md
├── requirements.txt
├── LICENSE
├── hhl.py
├── post_processor.py
├── run.py
├── configs
│   ├── elliptic_classical.yaml
│   ├── elliptic_hybrid.yaml
│   ├── heat_classical.yaml
│   └── heat_hybrid.yaml
├── old_runfiles
│   ├── run.py
│   └── run2.py
└── src
    ├── classical
    │   └── classical_solver.py
    ├── quantum
    │   ├── qlsa_solver.py
    │   ├── swap_test.py
    │   └── spectral_gradient.py
    ├── models
    │   ├── elliptic_model.py
    │   └── heat_model.py
    ├── optimization
    │   └── optimizer.py
    └── experiments
        ├── elliptic_experiment.py
        └── heat_experiment.py
```


## Environment Setup

This project uses Python and several scientific computing and quantum simulation libraries.  
We recommend using a **Conda environment** to ensure all dependencies are installed correctly.

### 1. Create a Conda Environment

Create a new environment with Python 3.10:

```bash
conda create -n hpdeco python=3.10
```

### 2. Activate the Environment

```bash
conda activate hpdeco
```
<!--- pipreqs . --force --mode no-pin --->
### 3. Install Project Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```


### 4. Clone the QLSA Repository

This project relies on the **QLSA framework** for implementing quantum linear system algorithms such as the HHL algorithm.

Clone the repository:

```bash
git clone https://github.com/QCOL-LU/QLSAs.git
```

### 5. Install the Repository in Editable Mode

Navigate into the repository and install it in editable mode so that any modifications to the source code take effect immediately.

```bash
cd QLSAs
pip install -e .
```

Installing in editable mode allows Python to use the **local source files directly**, which is necessary because this project modifies the HHL implementation.

### 6. Replace the HHL Implementation

This project includes a modified `hhl.py` file. Replace the default implementation in the QLSA repository with the version provided in this repository.

Copy the file:

```bash
cp hhl.py QLSAs/src/qlsas/algorithms/hhl/hhl.py
```

Alternatively, manually replace the file located at:

```
QLSAs/src/qlsas/algorithms/hhl/hhl.py
```

with the `hhl.py` file included in this project.

### 7. Verify Installation

After completing the steps above, you can test the installation by running:

```bash
python run.py configs/heat_hybrid.yaml
```

If the installation is successful, the hybrid solver will run using the modified HHL implementation.


### Notes

- The hybrid solver relies on the **QLSA framework** for implementing the HHL algorithm.
- Quantum circuits are executed using **Qiskit simulators** .