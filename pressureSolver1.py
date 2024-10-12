import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import numba as nb
import matplotlib.pyplot as plt

@nb.jit(nopython=True)
def thomas_algorithm(l, d, u, r):
    """
    Solves a tridiagonal linear system of equations using the Thomas algorithm.

    :param l: Lower diagonal elements of the tridiagonal matrix (l[0] is unused).
    :param d: Main diagonal elements of the tridiagonal matrix.
    :param u: Upper diagonal elements of the tridiagonal matrix (u[-1] is unused).
    :param r: Right-hand side vector of the system.
    :return: Solution vector of the system.
    """
    solution = np.zeros_like(d)
    n = len(solution)

    # Forward elimination
    for k in range(1, n):
        xmult = l[k - 1] / d[k - 1]
        d[k] = d[k] - xmult * u[k - 1]
        r[k] = r[k] - xmult * r[k - 1]

    # Back substitution
    solution[n - 1] = r[n - 1] / d[n - 1]
    for k in range(n - 2, -1, -1):
        solution[k] = (r[k] - u[k] * solution[k + 1]) / d[k]

    return solution


class PressureSolver:
    def __init__(self,
                 N,
                 dt,
                 total_time_in_days,
                 rw=0.318,
                 re=1000.0,
                 h=11.0,
                 phi=0.25,
                 mu=1.0,
                 ct=7.8e-6,
                 Q=1000.0,
                 k=500,
                 pi=4100.0,
                 solver='dense',
                 lazy=False):

        # Unit conversion factors (input units --> SI)
        self.ft_to_m_ = 0.3048
        self.psi_to_pa_ = 6894.75729
        self.day_to_sec_ = 24. * 60. * 60.
        self.bbl_to_m3_ = 0.1589873

        # Grid
        self.N_ = N
        self.rw_ = rw * self.ft_to_m_
        self.re_ = re * self.ft_to_m_
        self.h_ = h * self.ft_to_m_

        # Rock and fluid properties
        self.k_ = k * 1e-15 / 1.01325  # from mD to m^2
        self.phi_ = phi
        self.mu_ = mu * 1e-3  # from cP to Pa.s
        self.ct_ = ct / self.psi_to_pa_  # from 1/psi to 1/Pa

        # Initial and boundary conditions
        self.Q_ = Q * self.bbl_to_m3_ / self.day_to_sec_
        self.pi_ = pi * self.psi_to_pa_

        # Time control for simulation
        self.dt_ = dt * self.day_to_sec_
        self.total_time_sec_ = total_time_in_days * self.day_to_sec_

        # Create the grid in y, and grid spacing
        self.ye = np.log(self.re_ / self.rw_)
        self.yw = np.log(self.rw_ / self.rw_)
        self.dy = (self.ye - self.yw) / self.N_
        self.yi = np.arange(self.yw, self.ye, self.dy) + self.dy / 2.0
        self.r = self.rw_ * np.exp(self.yi)
        self.r_ft = self.r / self.ft_to_m_

        # Diffusivity coefficient η
        self.eta_ = self.k_ / (self.mu_ * self.phi_ * self.ct_)

        # Constant term alpha for boundary conditions
        self.alpha = self.Q_ * self.mu_ / (2 * np.pi * self.k_ * self.h_)


        # Constant term beta for boundary conditions
        self.beta_ = (self.Q_ * self.mu_ * self.dy) / (2 * np.pi * self.k_ * self.h_)

        # Pre-calculate the vector of ξ values
        self.xi_ = (self.eta_ * np.exp(-2 * self.yi) * self.dt_) / (self.rw_ ** 2 * self.dy ** 2)

        # Solver choice
        self.solver = solver

        # Lazy parameter for steady-state boundary condition
        self.lazy = lazy

    def create_time_dependent_matrix_A(self):
        A = np.zeros((self.N_, self.N_))
        for i in range(1, self.N_ - 1):
            A[i, i - 1] = -self.xi_[i]
            A[i, i] = 1 + 2 * self.xi_[i]
            A[i, i + 1] = -self.xi_[i]

        A[0, 0] = 1 + self.xi_[0]
        A[0, 1] = -self.xi_[0]
        A[-1, -2] = -self.xi_[-1]
        A[-1, -1] = 1 + 3 * self.xi_[-1]

        self.A_time_dependent = A  # Save time-dependent A as an attribute

    def create_time_dependent_vector_d(self):
        d = np.zeros(self.N_)
        d[0] = -self.beta_ * self.xi_[0]
        d[-1] = 2 * self.pi_ * self.xi_[-1]
        self.d_time_dependent = d  # Save time-dependent d as an attribute

    def solve_system(self, A, d, p):
        if self.solver == 'dense':
            return scipy.linalg.solve(A, p + d)
        elif self.solver == 'sparse':
            A_sparse = scipy.sparse.csr_matrix(A)
            return scipy.sparse.linalg.spsolve(A_sparse, p + d)
        elif self.solver == 'thomas':
            # Extract diagonals for the Thomas algorithm
            lower_diag = np.diag(A, k=-1).copy()
            main_diag = np.diag(A).copy()
            upper_diag = np.diag(A, k=1).copy()
            rhs = (p + d).copy()
            return thomas_algorithm(lower_diag, main_diag, upper_diag, rhs)
        else:
            raise ValueError("Solver type must be 'dense', 'sparse', or 'thomas'.")
            
    def calculate_well_pressure(self, p):
        # p_0 is the pressure at the well block (first element in the array)
        p_0 = p[0]
        # Apply the derived formula for p_w
        p_w = p_0 - (self.dy / 2) * self.alpha
        return p_w

    def solve_time_dependent(self):
        t_steps = int(self.total_time_sec_ / self.dt_) + 1
        self.p_array = np.zeros((t_steps + 1, self.N_))
        p = np.full(self.N_, self.pi_)
        self.p_array[0] = p.copy()

        self.create_time_dependent_matrix_A()
        self.create_time_dependent_vector_d()

        # Initialize well_pressures attribute to store pressures at each time step
        self.well_pressures = np.zeros(t_steps + 1)
        self.well_pressures[0] = self.calculate_well_pressure(p)
        
        for n in range(t_steps):
            p = self.solve_system(self.A_time_dependent, self.d_time_dependent, p)
            self.p_array[n + 1] = p.copy()
            self.well_pressures[n + 1] = self.calculate_well_pressure(p)

    def create_steady_state_A_d(self):
        """
        Creates the steady-state matrix A and vector d based on the provided code.
        Uses the 'lazy' parameter to determine the boundary condition at the reservoir boundary.
        """
        N = self.N_
        p_init = self.pi_
        alpha = 1 #self.alpha
        dy = self.dy

        # Initialize vectors
        a = np.ones(N - 1)
        b_diag = np.repeat(-2, N)
        c = np.ones(N - 1)
        d = np.zeros(N)

        # Boundary conditions
        b_diag[0] = -1
        d[0] = alpha * dy
        d[-1] = -p_init
        if not self.lazy:
            b_diag[-1] = -3
            d[-1] = -2 * p_init

        # Create matrix A
        A = np.diag(a, k=-1) + np.diag(b_diag, k=0) + np.diag(c, k=1)

        # Save the steady-state A and d as attributes
        self.A_steady = A
        self.d_steady = d

    def solve_steady_state(self):
        self.create_steady_state_A_d()
        self.steay_state_p = np.linalg.solve(self.A_steady,self.d_steady)

    def analytical_solution(self):
        self.analytical_p = self.pi_ + 1 * (self.yi - self.ye) # self.pi_ +  self.alpha*(self.yi - self.ye)

    def calculate_error(self):
        self.solve_steady_state()
        self.analytical_solution()
        self.error = np.abs(self.steay_state_p[0]  - self.analytical_p[0])

'''
test_solver = PressureSolver(N=4, dt=1.0, total_time_in_days=1.0, lazy=True)
test_solver.create_steady_state_A_d()
print(test_solver.A_steady, test_solver.d_steady)

test_solver2 = PressureSolver(N=4, dt=1.0, total_time_in_days=1.0, lazy=False)
test_solver2.create_steady_state_A_d()
print(test_solver2.A_steady, test_solver2.d_steady)


''' 
# List of N values (number of grid points)
N_values = [100, 500, 1000]

errors_lazy = []
errors_not_lazy = []

for N in N_values:
    # For the lazy method
    solver_lazy = PressureSolver(N=N, dt=1.0, total_time_in_days=1.0, lazy=True)
    solver_lazy.calculate_error()
    errors_lazy.append(solver_lazy.error)
    
    # For the not lazy method
    solver_not_lazy = PressureSolver(N=N, dt=1.0, total_time_in_days=1.0, lazy=False)
    solver_not_lazy.calculate_error()
    errors_not_lazy.append(solver_not_lazy.error)

# Plotting the errors versus N
plt.figure(figsize=(10, 6))
plt.scatter(N_values, errors_lazy, label='Lazy method', marker='o')
plt.scatter(N_values, errors_not_lazy, label='Not Lazy method', marker='x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of grid points N')
plt.ylabel('Maximum Absolute Error')
plt.title('Numerical Error vs Grid Size')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
