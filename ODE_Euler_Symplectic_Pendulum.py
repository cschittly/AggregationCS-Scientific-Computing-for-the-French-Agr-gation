
import numpy as np
import matplotlib.pyplot as plt

def euler_explicit_scalar(a: float, y0: float, h: float, T: float):
    N = int(np.floor(T / h)) + 1
    t = np.linspace(0, N*h, N)
    y = np.zeros(N)
    y[0] = y0
    for n in range(N-1):
        y[n+1] = y[n] - h * a * y[n]
    return t, y

def euler_implicit_scalar(a: float, y0: float, h: float, T: float):
    N = int(np.floor(T / h)) + 1
    t = np.linspace(0, N*h, N)
    y = np.zeros(N)
    y[0] = y0
    denom = 1.0 + h * a
    for n in range(N-1):
        y[n+1] = y[n] / denom
    return t, y

def scalar_demo(a=1.0, h_list=(0.1, 0.5, 1.0, 2.0), T=10.0, y0=1.0):
    for h in h_list:
        tE, yE = euler_explicit_scalar(a, y0, h, T)
        tI, yI = euler_implicit_scalar(a, y0, h, T)
        t_fine = np.linspace(0, T, 400)
        y_true = y0 * np.exp(-a * t_fine)

        plt.figure()
        plt.plot(tE, yE, 'o-', label="Euler explicite")
        plt.plot(tI, yI, 's-', label="Euler implicite")
        plt.plot(t_fine, y_true, '--', label="Solution exacte")
        plt.xlabel('t'); plt.title(f"y' = -a y, a={a}, h={h}")
        plt.legend(); plt.show()

scalar_demo()



import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

@dataclass
class PendulumParams:
    g: float = 9.81
    ell: float = 1.0

def H(theta, p, par: PendulumParams):
    return (p**2) / (2.0 * par.ell**2) + par.g * par.ell * (1.0 - np.cos(theta))

def dH_dtheta(theta, p, par: PendulumParams):
    return par.g * par.ell * np.sin(theta)

def dH_dp(theta, p, par: PendulumParams):
    return p / (par.ell**2)

def symplectic_euler(theta0, p0, h, T, par: PendulumParams):
    N = int(np.floor(T / h)) + 1
    t = np.arange(N) * h  # t in [0, T]
    theta = np.zeros(N); p = np.zeros(N)
    theta[0], p[0] = theta0, p0
    for n in range(N-1):
        p[n+1]     = p[n] - h * dH_dtheta(theta[n], p[n], par)
        theta[n+1] = theta[n] + h * dH_dp(theta[n], p[n+1], par)
    return t, theta, p

def f_pendulum(t, Y, par: PendulumParams):
    theta, p = Y
    return np.array([ dH_dp(theta, p, par), -dH_dtheta(theta, p, par) ])

def pendulum_demo(theta0=np.pi/4, p0=0.0, h=0.2, T=50.0):
    par = PendulumParams()
    N = int(np.floor(T / h)) + 1
    t = np.arange(N) * h
    T_end = t[-1]

    t_s, th, p = symplectic_euler(theta0, p0, h, T_end, par)
    sol = solve_ivp(lambda tt, yy: f_pendulum(tt, yy, par),
                    (0.0, T_end), np.array([theta0, p0]), method='RK45',
                    t_eval=t)

    plt.figure(); plt.plot(t_s, th, '-', label=r'$\theta$ (sympl.)')
    plt.plot(t_s, p, '-', label=r'$p$ (sympl.)')
    plt.plot(sol.t, sol.y[0], '--', label=r'$\theta$ (RK45)')
    plt.plot(sol.t, sol.y[1], '--', label=r'$p$ (RK45)')
    plt.xlabel('t'); plt.title(f'Pendule — h={h}'); plt.legend(); plt.show()

    plt.figure(); plt.plot(th, p, '-', label='Symplectic Euler')
    plt.plot(sol.y[0], sol.y[1], '--', label='RK45')
    plt.xlabel(r'$\theta$'); plt.ylabel(r'$p$')
    plt.title('Portrait de phase'); plt.legend(); plt.show()

    H_sym = H(th, p, par); H_rk = H(sol.y[0], sol.y[1], par)
    plt.figure(); plt.plot(t_s, H_sym, '-', label='Symplectic Euler')
    plt.plot(sol.t, H_rk, '--', label='RK45')
    plt.xlabel('t'); plt.ylabel('H(θ,p)')
    plt.title('Énergie'); plt.legend(); plt.show()

pendulum_demo(h=0.2, T=50.0)
pendulum_demo(h=0.5, T=50.0)
