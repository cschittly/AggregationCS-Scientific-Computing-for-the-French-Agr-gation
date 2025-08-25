#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advection–Diffusion–Reaction (Antigen–Antibody) in 1D
-----------------------------------------------------

We solve on x in [0, L], t in [0, T]:

    ∂t C + (u/w) ∂x C = (v/w) ∂xx C - (k/w) S C
    ∂t S              =            - p k C S

C(x,t): antigen concentration
S(x,t): antibody concentration

- w:   "volume/porosity" scaling (dimensionless here)
- u:   advection speed (>0 means flow from left to right)
- v:   diffusion coefficient
- k,p: reaction parameters (reaction rate k, interaction factor p)

BCs/ICs (used in this script):
  - Left boundary (x=0): Dirichlet C(0,t) = c_d(t)
  - Right boundary (x=L): Neumann  ∂x C(L,t) = 0  (implemented as C_M = C_{M-1})
  - Initial conditions: C(x,0)=0,  S(x,0)=s0 (constant)

Numerics
--------
We use **operator splitting**:

1) Advection–Diffusion (AD) step: explicit (upwind for advection + FTCS for diffusion)
   Stability (rough guideline for explicit AD):
       a = u*dt/(w*dx), b = v*dt/(w*dx^2), and require  2*b + |a| <= 1.

2) Reaction (R) step: **solved exactly, pointwise** (no time-step restriction from reaction).
   For the local system during reaction:
        dC/dt = -(k/w) S C
        dS/dt = - p k  C S
   There is an invariant  I = S - p w C  (constant during R step).
   Then with dt and I:
      - set alpha = (k/w) * I, beta = p * k
      - if alpha != 0:
            C_{n+1} = (C* e^{-alpha dt}) / (1 + (beta/alpha) C (1 - e^{-alpha dt}))
            S_{n+1} = I + p w C_{n+1}
        else:
            C_{n+1} = C / (1 + beta C dt)
            S_{n+1} = I + p w C_{n+1}

We provide two split schemes:
- Lie splitting  :     (AD) -> (R)
- Strang splitting: (R/2) -> (AD) -> (R/2)   (higher time accuracy)

What you get when you run this file:
- Spatial profiles C(x,T) and S(x,T) for a couple of (p,k) scenarios
- Time evolution of masses ∫C dx and ∫S dx (trapezoidal rule)
- Stability report for the AD step

Dependencies: numpy, matplotlib

Author: you :)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Problem setup / source term
# -----------------------------

def cd_default(t: float) -> float:
    """Left boundary concentration C(0,t). Feel free to customize."""
    # A smooth pulse: rises then decays (units-free)
    return t * np.exp(-0.5 * t)


# -----------------------------
# Utilities (stability & mass)
# -----------------------------

def stability_numbers(w: float, u: float, v: float, dx: float, dt: float) -> tuple[float, float, float]:
    """Return (a, b, CFL) where:
       a = u*dt/(w*dx),  b = v*dt/(w*dx^2),  CFL = 2*b + |a| (should be <= 1)."""
    a = u * dt / (w * dx)
    b = v * dt / (w * dx * dx)
    return a, b, 2.0 * b + abs(a)


def trapz_mass(X: np.ndarray, Y: np.ndarray) -> float:
    """∫ Y(x) dx on grid X using the trapezoidal rule."""
    return np.trapz(Y, X)


# -----------------------------
# Reaction step (exact, local)
# -----------------------------

def reaction_exact(C: np.ndarray, S: np.ndarray, dt: float, w: float, k: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact local solution for the reaction-only system over one time step dt.
    Works element-wise on vectors C, S.

    Invariant: I = S - p*w*C is constant during this substep.
    """
    I = S - p * w * C
    alpha = (k / w) * I      # could be zero
    beta = p * k

    Cn1 = np.empty_like(C)
    # exp(-alpha dt)
    E = np.exp(-alpha * dt)
    mask = np.abs(alpha) > 1e-12  # generic tolerance

    # General case (alpha != 0)
    Cn1[mask] = (C[mask] * E[mask]) / (1.0 + (beta / alpha[mask]) * C[mask] * (1.0 - E[mask]))
    # Limit case (alpha ~ 0)
    Cn1[~mask] = C[~mask] / (1.0 + beta * C[~mask] * dt)

    Sn1 = I + p * w * Cn1

    # safety clamp to avoid negative round-off
    Cn1 = np.clip(Cn1, 0.0, None)
    Sn1 = np.clip(Sn1, 0.0, None)
    return Cn1, Sn1


# -------------------------------------------
# Advection–diffusion explicit (upwind+FTCS)
# -------------------------------------------

def advection_diffusion_explicit(C: np.ndarray,
                                 dt: float, dx: float,
                                 w: float, u: float, v: float,
                                 left_value: float,
                                 right_bc: str = "neumann") -> np.ndarray:
    """
    One explicit AD step (no reaction), with upwind for advection (u>=0 assumed) and FTCS for diffusion.
    Boundaries:
      - left Dirichlet: C(0) = left_value
      - right: 'neumann' (dC/dx=0) or 'dirichlet0' (C=0)
    """
    M = C.size - 1
    Cn = C
    Cn1 = Cn.copy()

    a = u * dt / (w * dx)              # upwind for u>=0
    b = v * dt / (w * dx * dx)

    # interior nodes j = 1..M-1
    Cn1[1:M] = (Cn[1:M]
                - a * (Cn[1:M] - Cn[0:M-1])
                + b * (Cn[2:M+1] - 2.0 * Cn[1:M] + Cn[0:M-1]))

    # left Dirichlet
    Cn1[0] = left_value

    # right boundary
    if right_bc.lower() == "neumann":
        Cn1[M] = Cn1[M - 1]
    elif right_bc.lower() == "dirichlet0":
        Cn1[M] = 0.0
    else:
        raise ValueError("Unknown right_bc. Use 'neumann' or 'dirichlet0'.")

    # positivity (optional)
    Cn1 = np.clip(Cn1, 0.0, None)
    return Cn1


# -----------------------------
# Splitting time integrators
# -----------------------------

def step_lie_split(C: np.ndarray, S: np.ndarray,
                   dt: float, dx: float, w: float, u: float, v: float,
                   k: float, p: float, cd, t_next: float,
                   right_bc: str = "neumann") -> tuple[np.ndarray, np.ndarray]:
    """Lie splitting: (AD) then (R)."""
    # AD step
    C_star = advection_diffusion_explicit(C, dt, dx, w, u, v, left_value=cd(t_next), right_bc=right_bc)
    # R step (exact)
    C_next, S_next = reaction_exact(C_star, S, dt, w, k, p)
    return C_next, S_next


def step_strang_split(C: np.ndarray, S: np.ndarray,
                      dt: float, dx: float, w: float, u: float, v: float,
                      k: float, p: float, cd, t_curr: float,
                      right_bc: str = "neumann") -> tuple[np.ndarray, np.ndarray]:
    """Strang splitting: (R, dt/2) -> (AD, dt) -> (R, dt/2)."""
    # R half
    C_half, S_half = reaction_exact(C, S, 0.5 * dt, w, k, p)
    # AD full
    C_star = advection_diffusion_explicit(C_half, dt, dx, w, u, v, left_value=cd(t_curr + dt), right_bc=right_bc)
    # R half
    C_next, S_next = reaction_exact(C_star, S_half, 0.5 * dt, w, k, p)
    return C_next, S_next


# -----------------------------
# Solvers (time loops)
# -----------------------------

def solve_split(L: float, M: int, T: float, N: int,
                w: float, u: float, v: float,
                p: float, k: float, s0: float,
                cd=cd_default,
                method: str = "lie",
                right_bc: str = "neumann",
                report_stability: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the ADR system using operator splitting (Lie or Strang).
    Returns (X, times, C, S, masses) where:
        X     shape (M+1,)
        times shape (N+1,)
        C,S   shape (M+1, N+1)
        masses: dict with 'massC', 'massS' (shape (N+1,))
    """
    dx, dt = L / M, T / N
    X = np.linspace(0.0, L, M + 1)
    times = np.linspace(0.0, T, N + 1)

    a, b, cfl = stability_numbers(w, u, v, dx, dt)
    if report_stability:
        print(f"[AD stability] a={a:.3f}, b={b:.3f}, 2b+|a|={cfl:.3f}  (should be ≤ 1 for explicit AD)")

    C = np.zeros((M + 1, N + 1))
    S = np.ones((M + 1, N + 1)) * s0

    # mass arrays
    massC = np.zeros(N + 1)
    massS = np.zeros(N + 1)
    massC[0] = trapz_mass(X, C[:, 0])
    massS[0] = trapz_mass(X, S[:, 0])

    # choose stepping function
    if method.lower() == "lie":
        stepper = lambda c, s, n: step_lie_split(
            c, s, dt, dx, w, u, v, k, p, cd, times[n + 1], right_bc
        )
    elif method.lower() == "strang":
        stepper = lambda c, s, n: step_strang_split(
            c, s, dt, dx, w, u, v, k, p, cd, times[n], right_bc
        )
    else:
        raise ValueError("Unknown method. Use 'lie' or 'strang'.")

    # time loop
    for n in range(N):
        C[:, n + 1], S[:, n + 1] = stepper(C[:, n], S[:, n], n)
        massC[n + 1] = trapz_mass(X, C[:, n + 1])
        massS[n + 1] = trapz_mass(X, S[:, n + 1])

    masses = {"massC": massC, "massS": massS}
    return X, times, C, S, masses


# -----------------------------
# Plot helpers
# -----------------------------

def plot_profiles(X: np.ndarray, C: np.ndarray, S: np.ndarray, t_label: str, title: str):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(X, S, label="S at final time")
    plt.plot(X, C, label="C at final time")
    plt.xlabel("x (tube length)")
    plt.ylabel("concentration")
    plt.title(title + f"  ({t_label})")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_masses(times: np.ndarray, masses: dict, title: str):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(times, masses["massC"], label=r"$\int C\,dx$")
    plt.plot(times, masses["massS"], label=r"$\int S\,dx$")
    plt.xlabel("time")
    plt.ylabel("mass")
    plt.title(title + " — total masses over time")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()


# -----------------------------
# Demo / scenarios
# -----------------------------

def demo():
    # Common parameters (feel free to tweak)
    s0 = 10.0   # initial antibody concentration
    w  = 0.9
    u  = 0.4
    v  = 0.01
    L  = 4.0
    M  = 200    # space steps  -> dx = L/M = 0.02
    T  = 10.0
    N  = 1000   # time steps   -> dt = T/N = 0.01
    method = "strang"   # "lie" or "strang"

    scenarios = [
        # (p, k, label)
        (5.0, 70.0,  "p=5, k=70  (stiff reaction)"),
        (10.0, 0.5,  "p=10, k=0.5 (milder reaction)"),
    ]

    for p, k, label in scenarios:
        print(f"\n=== Running {label} [{method} splitting] ===")
        X, times, C, S, masses = solve_split(
            L, M, T, N, w, u, v, p, k, s0,
            cd=cd_default, method=method, right_bc="neumann", report_stability=True
        )
        # final time profiles
        plot_profiles(X, C[:, -1], S[:, -1], t_label=f"t = {times[-1]:.2f}",
                      title=f"ADR (antigen–antibody), {label}")
        # mass evolution
        plot_masses(times, masses, title=label)

    plt.show()


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    demo()
