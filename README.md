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
├── run.py
├── configs
│   ├── heat_classical.yaml
│   └── heat_hybrid.yaml
├── src
│   ├── classical
│   │   └── classical_solver.py
│   ├── quantum
│   │   ├── qlsa_solver.py
│   │   ├── swap_test.py
│   │   └── spectral_gradient.py
│   ├── models
│   │   └── heat_model.py
│   ├── optimization
│   │   └── optimizer.py
│   └── experiments
│       └── heat_experiment.py
```