import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pressureSolver import PressureSolver

# Load the data file
data = np.loadtxt('well_bhp.dat', skiprows=1)
time_minutes = data[:, 0]
pressure_psi = data[:, 1]

# Convert time from minutes to days and pressure from psi to pascal
time_days = time_minutes / (24 * 60)
pressure_pa = pressure_psi * 6894.75729

# Define the objective function
def objective(params):
    # Unpack parameters
    k, pi, re = params
    
    # Initialize the PressureSolver with the given parameters
    solver = PressureSolver(
        N=100,  # You can adjust the grid points if needed
        dt=0.01,  # Time step in days
        total_time_in_days=np.max(time_days),  # Use the max time from data
        k=k, 
        pi=pi,
        re=re,
        solver='sparse'  # Use sparse solver for fitting
    )
    
    # Solve the model
    solver.solve_time_dependent()
    
    # Convert model time to days for comparison
    model_time_days = np.arange(0, len(solver.well_pressures)) * solver.dt_ / solver.day_to_sec_
    well_pressure_psi = solver.well_pressures / 6894.75729  # Convert Pa to psi

    # Ensure length consistency between time and pressure arrays
    if len(model_time_days) != len(well_pressure_psi):
        print("Length mismatch between time and pressure arrays.")
        return np.inf

    # Interpolate to get model pressures at the observed times
    model_pressure_interpolated = np.interp(time_days, model_time_days, well_pressure_psi)
    
    # Calculate the error (RMSE)
    error = np.sqrt(np.mean((model_pressure_interpolated - pressure_psi) ** 2))
    return error

# Initial guesses for parameters
initial_guess = [500, 4100, 1000]

# Perform the optimization
result = opt.minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(100, 2000), (1000, 5000), (500, 3000)])
best_k, best_pi, best_re = result.x

print(f"Optimized k: {best_k}, Optimized pi: {best_pi}, Optimized re: {best_re}")

# Re-run the model with optimized parameters
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
plt.title('Fit of Numerical Model to Observed Well Pressure')
plt.grid(True)
plt.show()
