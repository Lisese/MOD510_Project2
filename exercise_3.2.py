import numpy as np
import time
import matplotlib.pyplot as plt
from pressureSolver import PressureSolver

# Assume the PressureSolver class and thomas_algorithm function have been defined as provided.

def measure_time(N, solver, dt=0.1, total_time_in_days=1):
    # Create an instance of PressureSolver
    ps = PressureSolver(N=N, dt=dt, total_time_in_days=total_time_in_days, solver=solver)
    # Warm-up run to compile any numba functions
    p_array = ps.solve_time_dependent()
    # Measure the time taken to solve the time-dependent problem
    start_time = time.time()
    p_array = ps.solve_time_dependent()
    end_time = time.time()
    return end_time - start_time

# Define N values to test (logarithmically spaced)
N_values = [100, 200, 400, 800, 1600, 3200, 6400]

# Define solvers to test
solvers = ['dense', 'sparse', 'thomas']

# Prepare to store times
times = {solver: [] for solver in solvers}

# Loop over N_values and solvers, and measure times
for N in N_values:
    print(f"Running for N = {N}")
    for solver in solvers:
        print(f"  Solver: {solver}")
        t = measure_time(N=N, solver=solver)
        times[solver].append(t)
        print(f"    Time taken: {t:.4f} seconds")

# Plot the results
for solver in solvers:
    plt.plot(N_values, times[solver], marker='o', label=solver)

plt.xlabel('N (number of grid points)')
plt.ylabel('Execution time (seconds)')
plt.title('Solver Performance Comparison')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.show()
