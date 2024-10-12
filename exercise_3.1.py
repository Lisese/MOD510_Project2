import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from pressureSolver import PressureSolver

# Initialize the solver
solver = PressureSolver(N=50, dt=0.01, total_time_in_days=1)  # Higher resolution for comparison

# Extract constants from the solver instance
Q = solver.Q_          # Flow rate (converted to SI units within the class)
mu = solver.mu_        # Viscosity (in Pa.s)
h = solver.h_          # Reservoir height (in meters)
k = solver.k_          # Permeability (in m^2)
pi = solver.pi_        # Initial pressure (in Pa)
eta = solver.eta_      # Hydraulic diffusivity

# Define the line-source solution function
def line_source_solution(r, t):
    term1 = (Q * mu) / (4 * np.pi * k * h)
    W_term = expi(-r**2 / (4 * eta * t))
    return pi + term1 * W_term

# Solve for the numerical solution over time
numerical_solution = solver.solve_time_dependent()

# Define radius values in meters (from solver) and times in days to plot
r_values = solver.r  # Radius in meters
t_values = [0.1, 0.2, 0.5, 0.8]  # in days
dt_days = solver.dt_ / solver.day_to_sec_  # Convert dt back to days

# Plotting
plt.figure(figsize=(12, 6))

for t in t_values:
    # Convert time to seconds
    t_seconds = t * 24 * 3600
    
    # Calculate line-source solution at the specified times
    p_line_source = line_source_solution(r_values, t_seconds)
    p_line_source_psi = p_line_source / solver.psi_to_pa_  # Convert Pa to psi for comparison

    # Get numerical solution at corresponding time step
    n = int(t / dt_days)
    p_numerical_psi = numerical_solution[n, :] / solver.psi_to_pa_  # Convert Pa to psi

    # Plot both solutions
    plt.plot(solver.r_ft, p_numerical_psi, 'o-', label=f'Numerical t={t} days')
    plt.plot(solver.r_ft, p_line_source_psi, '--', label=f'Line Source t={t} days')

# Set labels and title
plt.xlabel('Radius (ft)')
plt.ylabel('Pressure (psi)')
plt.title('Comparison of Numerical and Line-Source Solutions')
plt.legend()
plt.grid(True)
plt.show()
