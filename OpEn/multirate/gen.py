import sys
sys.path.append('..')

import casadi.casadi as cs
import opengen as og
from optimizer_base import CartPoleDynamics, CostFunction, ConstraintsBuilder, OptimizerGenerator


class MultiRateOptimizer(OptimizerGenerator):
    """Multi-rate NMPC optimizer generator."""

    def __init__(self, sampling_time=0.1, horizon_time=1.0):
        dynamics = CartPoleDynamics()
        super().__init__(dynamics)
        self.sampling_time = sampling_time
        self.horizon_time = horizon_time
        self.T = int(horizon_time / sampling_time)

    def generate(self):
        """Generate the multi-rate optimizer."""
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
        umin = [-20] * self.T
        umax = [20] * self.T
        bounds = og.constraints.Rectangle(umin, umax)

        # State constraints
        constraints_builder = ConstraintsBuilder(self.dynamics, self.sampling_time, self.T)
        f2 = constraints_builder.state_constraints_pm(x0, u_seq, state_bounds)

        # Create problem
        problem = og.builder.Problem(optimization_variables, optimization_parameters, total_cost) \
            .with_constraints(bounds) \
            .with_penalty_constraints(f2)

        # Build optimizer
        build_dir_name = f"python_cartpole_{self.sampling_time}"
        opt_name = f"cartpole_{self.sampling_time}"
        solver_config = self.create_solver_config()
        self.build_optimizer(problem, build_dir_name, opt_name, solver_config)


if __name__ == "__main__":
    # Change sampling_time to 0.1, 0.25, or 0.5 as needed
    optimizer = MultiRateOptimizer(sampling_time=0.1, horizon_time=1.0)
    optimizer.generate()
