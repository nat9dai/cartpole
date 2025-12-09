import casadi.casadi as cs
import opengen as og

class CartPoleDynamics:
    """CartPole dynamics for CasADi-based optimization."""

    def __init__(self, gravity_acceleration=9.8, length=1.0, mass_cart=1.0, mass_pole=0.2, dim=4):
        self.gravity_acceleration = gravity_acceleration
        self.length = length
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.dim = dim

    def dynamics_ct(self, x, u):
        """Continuous-time dynamics."""
        dx1 = x[2]
        dx2 = x[3]
        dx3 = (-self.mass_pole * self.length * cs.sin(x[1]) * x[3]**2
               + self.mass_pole * self.gravity_acceleration * cs.sin(x[1]) * cs.cos(x[1])
               + u) / (self.mass_cart + self.mass_pole * cs.sin(x[1])**2)
        dx4 = (-self.mass_pole * self.length * cs.sin(x[1]) * cs.cos(x[1]) * x[3]**2
               + (self.mass_cart + self.mass_pole) * self.gravity_acceleration * cs.sin(x[1])
               + u * cs.cos(x[1])) / (self.length * (self.mass_cart + self.mass_pole * cs.sin(x[1])**2))
        return [dx1, dx2, dx3, dx4]

    def dynamics_dt(self, x, u, h):
        """Discrete-time dynamics using RK45."""
        k1 = self.dynamics_ct(x, u)

        x_k2 = [x[i] + 0.5 * h * k1[i] for i in range(self.dim)]
        k2 = self.dynamics_ct(x_k2, u)

        x_k3 = [x[i] + 0.5 * h * k2[i] for i in range(self.dim)]
        k3 = self.dynamics_ct(x_k3, u)

        x_k4 = [x[i] + h * k3[i] for i in range(self.dim)]
        k4 = self.dynamics_ct(x_k4, u)

        x_next = [x[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(self.dim)]
        return x_next


class CostFunction:
    """Cost functions for MPC optimization."""

    @staticmethod
    def stage_cost(x, u, Q, R):
        """Stage cost: quadratic cost on state and control."""
        cost = Q[0] * x[0]**2 + Q[1] * x[1]**2 + Q[2] * x[2]**2 + Q[3] * x[3]**2 + R * u**2
        return cost

    @staticmethod
    def stage_cost_state_only(x, Q):
        """Stage cost with only state (no control)."""
        cost = Q[0] * x[0]**2 + Q[1] * x[1]**2 + Q[2] * x[2]**2 + Q[3] * x[3]**2
        return cost

    @staticmethod
    def terminal_cost(x, Q):
        """Terminal cost: quadratic cost on final state."""
        cost = Q[0] * x[0]**2 + Q[1] * x[1]**2 + Q[2] * x[2]**2 + Q[3] * x[3]**2
        return cost


class ConstraintsBuilder:
    """Builds state constraints for optimization."""

    def __init__(self, dynamics, sampling_time, T):
        self.dynamics = dynamics
        self.sampling_time = sampling_time
        self.T = T

    def state_constraints_alm(self, x, u, x_min=-7.0, x_max=7.0, x_dot_min=-15.0, x_dot_max=15.0):
        """Augmented Lagrangian method constraints."""
        c_min = [x_min, x_dot_min] * self.T
        c_max = [x_max, x_dot_max] * self.T
        f1 = []
        x_t = x

        for i in range(self.T):
            f1 = cs.vertcat(f1, x_t[0])
            f1 = cs.vertcat(f1, x_t[2])
            x_t = self.dynamics.dynamics_dt(x_t, u[i], self.sampling_time)

        C = og.constraints.Rectangle(c_min, c_max)
        return f1, C

    def state_constraints_pm(self, x, u, bounds):
        """Penalty method constraints."""
        x_min = bounds[0]
        x_max = bounds[1]
        x_dot_min = bounds[2]
        x_dot_max = bounds[3]
        f2 = []
        x_t = x

        for i in range(self.T):
            f2 = cs.vertcat(f2, cs.fmax(0, x_t[0] - x_max))
            f2 = cs.vertcat(f2, cs.fmax(0, x_min - x_t[0]))
            f2 = cs.vertcat(f2, cs.fmax(0, x_t[2] - x_dot_max))
            f2 = cs.vertcat(f2, cs.fmax(0, x_dot_min - x_t[2]))
            x_t = self.dynamics.dynamics_dt(x_t, u[i], self.sampling_time)

        return f2

    def state_constraints_pm_lifted(self, x, u, bounds, N):
        """Penalty method constraints for lifted formulation."""
        x_min = bounds[0]
        x_max = bounds[1]
        x_dot_min = bounds[2]
        x_dot_max = bounds[3]
        f2 = []
        x_t = x

        for i in range(self.T):
            for _ in range(N):
                f2 = cs.vertcat(f2, cs.fmax(0, x_t[0] - x_max))
                f2 = cs.vertcat(f2, cs.fmax(0, x_min - x_t[0]))
                f2 = cs.vertcat(f2, cs.fmax(0, x_t[2] - x_dot_max))
                f2 = cs.vertcat(f2, cs.fmax(0, x_dot_min - x_t[2]))
                x_t = self.dynamics.dynamics_dt(x_t, u[i], self.sampling_time / N)

        return f2


class OptimizerGenerator:
    """Base class for generating OpEn optimizers."""

    def __init__(self, dynamics, dim=4):
        self.dynamics = dynamics
        self.dim = dim
        self.cost_fn = CostFunction()

    def create_symbolic_variables(self, T):
        """Create symbolic variables for optimization."""
        u_seq = cs.MX.sym("u", T)
        x0 = cs.MX.sym("x0", self.dim)
        Q = cs.MX.sym("Q", self.dim)
        Qt = cs.MX.sym("Qt", self.dim)
        R = cs.MX.sym("R", 1)
        state_bounds = cs.MX.sym("state_bounds", 4)
        return u_seq, x0, Q, Qt, R, state_bounds

    def build_optimizer(self, problem, build_dir_name, opt_name, solver_config):
        """Build the optimizer using OpEn."""
        build_config = og.config.BuildConfiguration() \
            .with_build_directory(build_dir_name.replace(".", "_")) \
            .with_tcp_interface_config()

        meta = og.config.OptimizerMeta().with_optimizer_name(opt_name.replace(".", "_"))

        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
        builder.build()

    def create_solver_config(self, tolerance=1e-6, initial_tolerance=1e-6, max_duration_micros=450000):
        """Create solver configuration."""
        return og.config.SolverConfiguration() \
            .with_tolerance(tolerance) \
            .with_initial_tolerance(initial_tolerance) \
            .with_max_duration_micros(max_duration_micros)
