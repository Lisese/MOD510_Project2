import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pressureSolver import PressureSolver
# Load the data file
data = np.loadtxt('well_bhp.dat', skiprows=1)
time_minutes = data[:, 0]
pressure_psi = data[:, 1]

# Convert time from minutes to days for the model
time_days = time_minutes / (24 * 60)

# Define the model function for curve fitting
def model_function(time_days, k, pi, re):
    # Initialize the PressureSolver with the given parameters
    solver = PressureSolver(
        N=100,  # Adjust grid points if needed
        dt=0.01,  # Time step in days
        total_time_in_days=np.max(time_days),  # Use the max time from data
        k=k, 
        pi=pi,
        re=re,
        solver='sparse'  # Use sparse solver for fitting
    )
    
    # Solve the model
    solver.solve_time_dependent()
    
    # Convert well pressure to psi
    well_pressure_psi = solver.well_pressures / 6894.75729  # Convert Pa to psi
    
    # Interpolate to get model pressures at the observed times
    model_time_days = np.arange(0, len(solver.well_pressures)) * solver.dt_ / solver.day_to_sec_
    model_pressure_interpolated = np.interp(time_days, model_time_days, well_pressure_psi)
    
    return model_pressure_interpolated

# Initial guesses for k, pi, and re
initial_guess = [500, 4100, 1000]

# Use curve_fit to find the best parameters
params_opt, params_cov = opt.curve_fit(model_function, time_days, pressure_psi, p0=initial_guess, bounds=([100, 1000, 500], [2000, 5000, 2000]))

# Extract optimized parameters
best_k, best_pi, best_re = params_opt
print(f"Optimized k: {best_k}, Optimized pi: {best_pi}, Optimized re: {best_re}")

# Re-run the model with optimized parameters for plotting
optimized_solver = PressureSolver(
    N=100, 
    dt=0.01, 
    total_time_in_days=np.max(time_days), 
    k=best_k, 
    pi=best_pi,
    re=best_re,
    solver='sparse'
)
optimized_solver.solve_time_dependent()

# Convert well pressure to psi for plotting
well_pressure_psi = optimized_solver.well_pressures / 6894.75729  # Convert Pa to psi

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_minutes, pressure_psi, 'o', label='Observed Data')
plt.plot(time_minutes, well_pressure_psi[:len(time_minutes)], '-', label='Optimized Model Fit')
plt.xlabel('Time (minutes)')
plt.ylabel('Well Pressure (psi)')
plt.legend()
plt.title('Curve Fit of Numerical Model to Observed Well Pressure')
plt.grid(True)
plt.show()
