import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
from IPython.display import display, Math, Latex

# Parameters
r = 2
rho = 1

# Original function (seems unused in main analysis)
def funct(t, x):
    return -r*x + np.sin(t)

# Utility functions
def U_orig(x):
    """Original utility function"""
    return 5*np.log(1+x)

def der_U_orig(x):
    """Derivative of original utility function"""
    return 5/(1+x)

def D(c):
    """Damage function - Fixed syntax error"""
    return 0.5*(c-0.2)**2  # Fixed: ** instead of ^

def der_D(c):
    """Derivative of damage function"""
    return c-0.2

def iso_D(x):
    """Isocline for D"""
    return der_D(r*x)

def iso_U_func(x):
    """Isocline for U - renamed to avoid conflict"""
    return der_U_orig(x)/(r+rho)

def phi(l):
    """Reciprocal function"""
    return l+0.2

def der_phi(l):
    """Derivative of phi"""
    return 1

# System equations - renamed to avoid conflicts
def U_system(x, l):
    """dx/dt equation"""
    return -r*x + phi(l)

def V_system(x, l):
    """dl/dt equation"""
    return (r+rho)*l - der_U_orig(x)

# Differential equation system
def eq_diff_f(x, l):
    """System of equations"""
    f1 = U_system(x, l)
    f2 = V_system(x, l)
    return f1, f2

def eq_diff(t, y):
    """Differential equation in standard form for solve_ivp"""
    f = np.zeros(2)
    f[0], f[1] = eq_diff_f(y[0], y[1])
    return f

# Numerical methods
def Euler(f, y0, N, T):
    """Explicit Euler method"""
    tps = np.linspace(0, T, N+1)
    h = tps[1] - tps[0]
    n = y0.shape[0]
    y = np.zeros((N+1, n))
    y[0, :] = y0
    
    for j in range(1, N+1):
        y[j, :] = y[j-1, :] + h * f((j-1)*h, y[j-1, :])
    
    return tps, y

def RungeKutta(f, y0, N, T):
    """4th order Runge-Kutta method"""
    tps = np.linspace(0, T, N+1)
    h = tps[1] - tps[0]
    n = y0.shape[0]
    y = np.zeros((N+1, n))
    y[0, :] = y0
    
    for j in range(1, N+1):
        tj, yj = (j-1)*h, y[j-1, :]
        k1 = f(tj, yj)
        k2 = f(tj + h/2, yj + h/2 * k1)
        k3 = f(tj + h/2, yj + h/2 * k2)
        k4 = f(tj + h, yj + h * k3)
        y[j, :] = y[j-1, :] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return tps, y

# Analysis and plotting
def plot_phase_portrait():
    """Create phase portrait with isoclines"""
    plt.figure(figsize=(10, 8))
    
    # Isoclines
    mesh = np.linspace(0, 4, 50)
    plt.plot(mesh, iso_U_func(mesh), "r.", label='Isocline U', markersize=4)
    plt.plot(mesh, iso_D(mesh), "b.", label='Isocline D', markersize=4)
    
    # Phase portrait
    xmin, xmax = 0, 4
    ymin, ymax = 0, 4
    x = np.linspace(xmin, xmax, 20)
    y = np.linspace(ymin, ymax, 20)
    X, Y = np.meshgrid(x, y)
    
    # Direction field
    U_field = U_system(X, Y)
    V_field = V_system(X, Y)
    
    plt.quiver(X, Y, U_field, V_field, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('λ (lambda)')
    plt.title("Isoclines and Phase Portrait")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 3])
    plt.xlim([0, 3.5])
    plt.show()

def solve_and_plot_system():
    """Solve system using scipy and plot results"""
    T = 10.
    t_span = [0., T]
    y0 = np.array([3.0, 1.3])  # More reasonable initial conditions
    t_eval = np.linspace(0, T, 100)
    
    sol = solve_ivp(fun=eq_diff, t_span=t_span, y0=y0, method='RK45', t_eval=t_eval)
    
    plt.figure(figsize=(12, 5))
    
    # Time series
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], 'r-', label='x(t)', linewidth=2)
    plt.plot(sol.t, sol.y[1], 'b-', label='λ(t)', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Variables')
    plt.title('Time Evolution (scipy solve_ivp)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Trajectory in phase space
    plt.subplot(1, 2, 2)
    plt.plot(sol.y[0], sol.y[1], 'g-', linewidth=2, label='Trajectory')
    plt.plot(sol.y[0, 0], sol.y[1, 0], 'go', markersize=8, label='Start')
    plt.plot(sol.y[0, -1], sol.y[1, -1], 'ro', markersize=8, label='End')
    plt.xlabel('x')
    plt.ylabel('λ (lambda)')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_numerical_methods():
    """Compare Euler and Runge-Kutta methods"""
    T = 5.
    y0 = np.array([3.0, 1.5])  # More reasonable initial conditions
    N = 20  # Increased number of steps for better accuracy
    
    # Solve with both methods
    tpsE, yE = Euler(eq_diff, y0, N, T)
    tpsRK, yRK = RungeKutta(eq_diff, y0, N, T)
    
    # Also solve with scipy for comparison
    t_eval = np.linspace(0, T, 100)
    sol_ref = solve_ivp(fun=eq_diff, t_span=[0., T], y0=y0, method='RK45', t_eval=t_eval)
    
    # Plot time series comparison
    plt.figure(figsize=(15, 10))
    
    # Time series for x(t)
    plt.subplot(2, 2, 1)
    plt.plot(tpsE, yE[:, 0], 'ro-', label='Euler x(t)', markersize=4)
    plt.plot(tpsRK, yRK[:, 0], 'bo-', label='Runge-Kutta x(t)', markersize=4)
    plt.plot(sol_ref.t, sol_ref.y[0], 'k-', label='Reference (RK45)', linewidth=2, alpha=0.7)
    plt.xlabel('Time t')
    plt.ylabel('x(t)')
    plt.title('Comparison: x(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time series for λ(t)
    plt.subplot(2, 2, 2)
    plt.plot(tpsE, yE[:, 1], 'ro-', label='Euler λ(t)', markersize=4)
    plt.plot(tpsRK, yRK[:, 1], 'bo-', label='Runge-Kutta λ(t)', markersize=4)
    plt.plot(sol_ref.t, sol_ref.y[1], 'k-', label='Reference (RK45)', linewidth=2, alpha=0.7)
    plt.xlabel('Time t')
    plt.ylabel('λ(t)')
    plt.title('Comparison: λ(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase space trajectories - FIXED VERSION
    plt.subplot(2, 2, 3)
    plt.plot(yE[:, 0], yE[:, 1], 'ro-', label='Euler trajectory', markersize=4)
    plt.plot(yRK[:, 0], yRK[:, 1], 'bo-', label='Runge-Kutta trajectory', markersize=4)
    plt.plot(sol_ref.y[0], sol_ref.y[1], 'k-', label='Reference trajectory', linewidth=2, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('λ (lambda)')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error analysis
    plt.subplot(2, 2, 4)
    # Interpolate reference solution at Euler/RK time points
    x_ref_E = np.interp(tpsE, sol_ref.t, sol_ref.y[0])
    lambda_ref_E = np.interp(tpsE, sol_ref.t, sol_ref.y[1])
    x_ref_RK = np.interp(tpsRK, sol_ref.t, sol_ref.y[0])
    lambda_ref_RK = np.interp(tpsRK, sol_ref.t, sol_ref.y[1])
    
    error_E = np.sqrt((yE[:, 0] - x_ref_E)**2 + (yE[:, 1] - lambda_ref_E)**2)
    error_RK = np.sqrt((yRK[:, 0] - x_ref_RK)**2 + (yRK[:, 1] - lambda_ref_RK)**2)
    
    plt.semilogy(tpsE, error_E, 'ro-', label='Euler error', markersize=4)
    plt.semilogy(tpsRK, error_RK, 'bo-', label='Runge-Kutta error', markersize=4)
    plt.xlabel('Time t')
    plt.ylabel('Error (log scale)')
    plt.title('Error Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final Euler error: {error_E[-1]:.6f}")
    print(f"Final Runge-Kutta error: {error_RK[-1]:.6f}")
    print(f"RK improvement factor: {error_E[-1]/error_RK[-1]:.2f}x")

# Main execution
if __name__ == "__main__":
    print("=== Phase Portrait Analysis ===")
    plot_phase_portrait()
    
    print("\n=== System Solution with scipy ===")
    solve_and_plot_system()
    
    print("\n=== Numerical Methods Comparison ===")
    compare_numerical_methods()
