import numpy as np
from cartpole_model import CartPoleModel


class SimulationResult:
    """Container for simulation results."""

    def __init__(self):
        self.state_sequence = []
        self.u_sequence = []
        self.solve_time = []

    def append(self, state, control, solve_time):
        """Add a simulation step result."""
        self.state_sequence.append(state)
        self.u_sequence.append(control)
        self.solve_time.append(solve_time)

    def get_flat_states(self):
        """Flatten state sequence for plotting."""
        return [item for sublist in self.state_sequence for item in sublist]


class Simulator:
    """Simulates CartPole control with MPC."""

    def __init__(self, model, sampling_time_sim, simulation_steps):
        """
        Args:
            model: CartPoleModel instance
            sampling_time_sim: Simulation time step
            simulation_steps: Total number of simulation steps
        """
        self.model = model
        self.sampling_time_sim = sampling_time_sim
        self.simulation_steps = simulation_steps

    def run_single_rate(self, mng, x_state_0, Q, Qt, R, bounds, sampler_interval, horizon_length):
        """Run single-rate MPC simulation.

        Args:
            mng: Optimizer TCP manager
            x_state_0: Initial state
            Q: State cost weights
            Qt: Terminal state cost weights
            R: Control cost weights
            bounds: State bounds [x_min, x_max, x_dot_min, x_dot_max]
            sampler_interval: Sampling interval in simulation steps
            horizon_length: MPC horizon length

        Returns:
            SimulationResult object
        """
        x = x_state_0
        us = [0] * horizon_length
        result = SimulationResult()
        solve_time_one_step = 0.0

        for k in range(self.simulation_steps):
            if k % sampler_interval == 0:
                solver_status = mng.call(x + Q + Qt + R + bounds, initial_guess=us)
                us = solver_status['solution']
                u = us[0]
                solve_time_one_step = solver_status['solve_time_ms']

            x_next = self.model.dynamics_dt(x, u, sampling_time=self.sampling_time_sim)
            result.append(x_next, u, solve_time_one_step)
            x = x_next

        return result

    def run_multirate(self, mng, x_state_0, Q, Qt, R, bounds, sampler_interval,
                      control_interval, N_P, horizon_length):
        """Run multi-rate MPC simulation.

        Args:
            mng: Optimizer TCP manager
            x_state_0: Initial state
            Q: State cost weights
            Qt: Terminal state cost weights
            R: Control cost weights
            bounds: State bounds
            sampler_interval: Sampling interval in simulation steps
            control_interval: Control update interval in simulation steps
            N_P: Number of control inputs per sample period
            horizon_length: MPC horizon length

        Returns:
            SimulationResult object
        """
        x = x_state_0
        us = [0] * N_P * horizon_length
        u = 0.0
        result = SimulationResult()
        solve_time_one_step = 0.0
        count = 0
        control_sequence = [0] * N_P

        for k in range(self.simulation_steps):
            if k % sampler_interval == 0:
                solver_status = mng.call(x + Q + Qt + R + bounds, initial_guess=us)
                us = solver_status['solution']
                solve_time_one_step = solver_status['solve_time_ms']
                control_sequence = us[0:N_P]
                count = 0

            if k % control_interval == 0:
                u = control_sequence[count]
                count += 1

            x_next = self.model.dynamics_dt(x, u, sampling_time=self.sampling_time_sim)
            result.append(x_next, u, solve_time_one_step)
            x = x_next

        return result
