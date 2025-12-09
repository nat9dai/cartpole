import sys
sys.path.append('..')

import casadi.casadi as cs
import opengen as og
from optimizer_base import CartPoleDynamics, CostFunction, ConstraintsBuilder, OptimizerGenerator


class LiftedMultiRateMultiControlOptimizer(OptimizerGenerator):
    """Lifted multi-rate NMPC with multiple control inputs per sampling period."""

    def __init__(self, sampling_time=0.1, control_sampling_time=0.05, horizon_time=1.0, N=10):
        dynamics = CartPoleDynamics()
        super().__init__(dynamics)
        self.sampling_time = sampling_time
        self.control_sampling_time = control_sampling_time
        self.horizon_time = horizon_time
        self.N = N  # Number of subintervals for Simpson's rule
        self.N_P = int(sampling_time / control_sampling_time)  # Controls per sampling period
        self.T = int(horizon_time / sampling_time)

    def generate(self):
        """Generate the lifted multi-rate multi-control optimizer."""
        # Create symbolic variables with extended control sequence
        u_seq = cs.MX.sym("u", self.T * self.N_P)
        x0 = cs.MX.sym("x0", self.dim)
        Q = cs.MX.sym("Q", self.dim)
        Qt = cs.MX.sym("Qt", self.dim)
        R = cs.MX.sym("R", 1)
        state_bounds = cs.MX.sym("state_bounds", 4)

        # Precompute parameters for Simpson's rule
        sampling_div_2N = self.sampling_time / (2 * self.N)
        sampling_div_N = self.sampling_time / self.N

        # Compute total cost using Simpson's rule with multi-rate control
        x_t = x0
        total_cost = 0
        u_index = 0

        for i in range(self.T):
            for j in range(self.N):
                # Simpson's rule to approximate the integral (state cost only)
                integral = self.cost_fn.stage_cost_state_only(x_t, Q)
                x_t_mid = self.dynamics.dynamics_dt(x_t, u_seq[u_index], sampling_div_2N)
                integral += 4 * self.cost_fn.stage_cost_state_only(x_t_mid, Q)
                x_t = self.dynamics.dynamics_dt(x_t, u_seq[u_index], sampling_div_N)
                integral += self.cost_fn.stage_cost_state_only(x_t, Q)
                total_cost += integral / (6 * self.N)

                # Add control cost at appropriate intervals
                if (j % (self.N / self.N_P) == (self.N / self.N_P) - 1):
                    total_cost += (0.1 / self.N_P) * u_seq[u_index]**2
                    u_index += 1

        total_cost += self.cost_fn.terminal_cost(x_t, Qt)

        # Setup optimization variables and parameters
        optimization_variables = cs.vertcat(u_seq)
        optimization_parameters = cs.vertcat(x0, Q, Qt, R, state_bounds)

        # Control input bounds
        umin = [-20] * self.T * self.N_P
        umax = [20] * self.T * self.N_P
        bounds = og.constraints.Rectangle(umin, umax)

        # State constraints (lifted)
        constraints_builder = ConstraintsBuilder(self.dynamics, self.sampling_time, self.T)
        f2 = constraints_builder.state_constraints_pm_lifted(x0, u_seq, state_bounds, self.N)

        # Create problem
        problem = og.builder.Problem(optimization_variables, optimization_parameters, total_cost) \
            .with_constraints(bounds) \
            .with_penalty_constraints(f2)

        # Build optimizer
        build_dir_name = f"python_cartpole_lifting_multirate_{self.sampling_time}"
        opt_name = f"cartpole_lifting_multirate_{self.sampling_time}"
        solver_config = self.create_solver_config()
        self.build_optimizer(problem, build_dir_name, opt_name, solver_config)


if __name__ == "__main__":
    # Change sampling_time to 0.1, 0.25, or 0.5 as needed
    optimizer = LiftedMultiRateMultiControlOptimizer(
        sampling_time=0.1,
        control_sampling_time=0.05,
        horizon_time=1.0,
        N=10
    )
    optimizer.generate()
