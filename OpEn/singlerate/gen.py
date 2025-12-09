import sys
sys.path.append('..')

import casadi.casadi as cs
import opengen as og
from optimizer_base import CartPoleDynamics, CostFunction, ConstraintsBuilder, OptimizerGenerator


class SingleRateOptimizer(OptimizerGenerator):
    """Single-rate NMPC optimizer generator."""

    def __init__(self, T=20, sampling_time=0.05):
        dynamics = CartPoleDynamics()
        super().__init__(dynamics)
        self.T = T
        self.sampling_time = sampling_time

    def generate(self):
        """Generate the single-rate optimizer."""
        # Create symbolic variables
        u_seq, x0, Q, Qt, R, state_bounds = self.create_symbolic_variables(self.T)

        # Compute total cost
        x_t = x0
        total_cost = 0
        for t in range(self.T):
            total_cost += self.cost_fn.stage_cost(x_t, u_seq[t], Q, R)
            x_t = self.dynamics.dynamics_dt(x_t, u_seq[t], self.sampling_time)

        total_cost += self.cost_fn.terminal_cost(x_t, Qt)

        # Setup optimization variables and parameters
        optimization_variables = cs.vertcat(u_seq)
        optimization_parameters = cs.vertcat(x0, Q, Qt, R, state_bounds)

        # Control input bounds
        umin = [-15] * self.T
        umax = [15] * self.T
        bounds = og.constraints.Rectangle(umin, umax)

        # State constraints
        constraints_builder = ConstraintsBuilder(self.dynamics, self.sampling_time, self.T)
        f2 = constraints_builder.state_constraints_pm(x0, u_seq, state_bounds)

        # Create problem
        problem = og.builder.Problem(optimization_variables, optimization_parameters, total_cost) \
            .with_constraints(bounds) \
            .with_penalty_constraints(f2)

        # Build optimizer
        solver_config = self.create_solver_config()
        self.build_optimizer(problem, "python_cartpole", "cartpole", solver_config)


if __name__ == "__main__":
    optimizer = SingleRateOptimizer(T=20, sampling_time=0.05)
    optimizer.generate()
