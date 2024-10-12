import pandas as pd
import matplotlib.pyplot as plt
from pressureSolver import PressureSolver
import numpy as np
''' 
data = pd.read_csv('well_bhp.dat', delim_whitespace=True, header=None)

# Rename columns based on the header
data.columns = data.iloc[0]  # Assign the first row as header
data = data.drop(0)  # Remove the header row from the data

# Convert columns to numeric types
data['time'] = pd.to_numeric(data['time'])
data['well_pressure'] = pd.to_numeric(data['well_pressure'])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['time'], data['well_pressure'], color='b', marker='o')
plt.xlabel('Time (minutes)')
plt.ylabel('Well Pressure (psi)')
plt.title('Well Pressure vs Time')
plt.grid(True)
plt.show()
''' 
pascal_to_psi = 1 / 6894.75729
minutes_to_days = 1 / (24 * 60)

# Read the data file
file_path = 'well_bhp.dat'
data = pd.read_csv(file_path, delim_whitespace=True, header=None)
data.columns = data.iloc[0]
data = data.drop(0)
data['time'] = pd.to_numeric(data['time'])
data['well_pressure'] = pd.to_numeric(data['well_pressure'])

# Set up solver and solve
N = 500
dt = 0.01
total_time_minutes = 2000
total_time_days = total_time_minutes * minutes_to_days

solver = PressureSolver(N=N, dt=dt, total_time_in_days=total_time_days)
solver.solve_time_dependent()
well_pressures_analytical_psi = solver.well_pressures * pascal_to_psi
time_steps_minutes = np.linspace(0, total_time_minutes, len(well_pressures_analytical_psi))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['time'], data['well_pressure'], color='b', marker='o', label='Observed Data')
plt.plot(time_steps_minutes, well_pressures_analytical_psi, color='r', label='Simulated Analytical Solution')
plt.xlabel('Time (minutes)')
plt.ylabel('Well Pressure (psi)')
plt.title('Well Pressure vs Time')
plt.legend()
plt.grid(True)
plt.show()