# AggregationCS – Scientific Computing Notebooks

This repository was originally created as preparation material for the *concours de l’Agrégation de Mathématiques* (oral exam in *calcul scientifique*).

It gathers a variety of **numerical methods and applications**, each presented in a clean, didactic way through Python scripts and Jupyter notebooks.  
Although designed with the Agrégation in mind, these notebooks can be useful to **anyone interested in scientific computing**: students, teachers, or anyone who wants to explore classical numerical methods with practical examples.

---

## Topics covered

- **Numerical ODEs**: Euler schemes, Runge–Kutta, symplectic integration.  
- **PDEs**: finite-difference discretizations (Poisson, advection, diffusion).  
- **Optimization**: gradient descent, quadratic minimization.  
- **Applied models**: Lotka–Volterra predator–prey, pendulum dynamics, biological growth models.  
- **Control & phase portraits**: isoclines, stability, dynamical behavior.

---

## Repository content 

- `EDP_Cancer.py` — PDE model in biology (cancer growth).  
- `FixedPoint_Newton_TP.ipynb` — Fixed-point iteration & Newton’s method.  
- `Gradient_Descent_Demos.ipynb` — Gradient descent demonstrations.  
- `linear_advection_schemes.ipynb` — Linear advection schemes, CFL stability.  
- `ODE_Euler_Symplectic_Pendulum.py` — Symplectic Euler scheme for the pendulum.  
- `ode_exponential_lotka_volterra.ipynb` — Exponential ODE & Lotka–Volterra.  
- `optim_quadratic.ipynb` — Gradient methods on quadratic functions.  
- `phase_portrait_control_system.ipynb` — Control system phase portraits.  
- `Poisson1D_FD_TP.ipynb` — 1D Poisson finite-difference solver.

---

## Purpose

- Provide **didactic, ready-to-use demos** for oral presentation.  
- Showcase a **broad range of numerical methods** and their applications.  
- Offer a resource that can be extended and reused beyond the Agrégation context.

---

## Usage

Clone the repository and open the notebooks with Jupyter:

```bash
git clone https://github.com/your-username/AggregationCS.git
cd AggregationCS
jupyter notebook
