import sys
sys.path.append('..')

import casadi.casadi as cs
import opengen as og
from optimizer_base import CartPoleDynamics, CostFunction, ConstraintsBuilder, OptimizerGenerator


class LiftedSingleRateOptimizer(OptimizerGenerator):
    """Lifted single-rate NMPC optimizer generator."""

    def __init__(self, T=20, sampling_time=0.05, N=10):
        dynamics = CartPoleDynamics()
        super().__init__(dynamics)
        self.T = T
        self.sampling_time = sampling_time
        self.N = N  # Number of subintervals for Simpson's rule

    def generate(self):
        """Generate the lifted single-rate optimizer."""
        # Create symbolic variables
        u_seq, x0, Q, Qt, R, state_bounds = self.create_symbolic_variables(self.T)

        # Precompute parameters for Simpson's rule
        sampling_div_2N = self.sampling_time / (2 * self.N)
        sampling_div_N = self.sampling_time / self.N

        # Compute total cost using Simpson's rule
        x_t = x0
        total_cost = 0
        for i in range(self.T):
            for j in range(self.N):
                # Simpson's rule to approximate the integral
                integral = self.cost_fn.stage_cost(x_t, u_seq[i], Q, R)
                x_t_mid = self.dynamics.dynamics_dt(x_t, u_seq[i], sampling_div_2N)
                integral += 4 * self.cost_fn.stage_cost(x_t_mid, u_seq[i], Q, R)
                x_t = self.dynamics.dynamics_dt(x_t, u_seq[i], sampling_div_N)
                integral += self.cost_fn.stage_cost(x_t, u_seq[i], Q, R)
                total_cost += integral / (6 * self.N)

        total_cost += self.cost_fn.terminal_cost(x_t, Qt)

        # Setup optimization variables and parameters
        optimization_variables = cs.vertcat(u_seq)
        optimization_parameters = cs.vertcat(x0, Q, Qt, R, state_bounds)

        # Control input bounds
        umin = [-15] * self.T
        umax = [15] * self.T
        bounds = og.constraints.Rectangle(umin, umax)

        # State constraints (lifted)
        constraints_builder = ConstraintsBuilder(self.dynamics, self.sampling_time, self.T)
        f2 = constraints_builder.state_constraints_pm_lifted(x0, u_seq, state_bounds, self.N)

        # Create problem
        problem = og.builder.Problem(optimization_variables, optimization_parameters, total_cost) \
            .with_constraints(bounds) \
            .with_penalty_constraints(f2)

        # Build optimizer with shorter max duration
        solver_config = self.create_solver_config(max_duration_micros=45000)
        self.build_optimizer(problem, "python_cartpole_lifted", "cartpole_lifted", solver_config)


if __name__ == "__main__":
    optimizer = LiftedSingleRateOptimizer(T=20, sampling_time=0.05, N=10)
    optimizer.generate()
