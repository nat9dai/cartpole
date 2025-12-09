import numpy as np
from cartpole_model import CartPoleModel
from simulator import Simulator
from optimizer_manager import OptimizerManager
from results_analyzer import ResultsAnalyzer
from plotter import Plotter


def run_simulation_for_period(sampler_period, model, sampling_time_sim, simulation_steps,
                               sampling_time_control, x_state_0, Q, Qt, R, bounds):
    """Run all three simulation types for a given sampler period.

    Args:
        sampler_period: Sampling period
        model: CartPoleModel instance
        sampling_time_sim: Simulation time step
        simulation_steps: Total simulation steps
        sampling_time_control: Control update period
        x_state_0: Initial state
        Q, Qt, R: Cost weights
        bounds: State bounds

    Returns:
        Dict with simulation results for each controller type
    """
    N_P = int(sampler_period / sampling_time_control)
    horizon_time = 1.0
    T = int(horizon_time / sampler_period)
    control_interval = int(sampling_time_control / sampling_time_sim)
    sampler_interval = int(sampler_period / sampling_time_sim)

    # Determine port offset based on period
    port_offset = {0.1: 0, 0.25: 3, 0.5: 6}.get(sampler_period, 0)
    base_port = 8331 + port_offset

    # Setup optimizers
    opt_manager = OptimizerManager()
    tcp_name1 = f"OpEn/multirate/python_cartpole_{sampler_period}/cartpole_{sampler_period}"
    tcp_name2 = f"OpEn/multirate/python_cartpole_lifting_{sampler_period}/cartpole_lifting_{sampler_period}"
    tcp_name3 = f"OpEn/multirate/python_cartpole_lifting_multirate_{sampler_period}/cartpole_lifting_multirate_{sampler_period}"

    mng1 = opt_manager.add_optimizer(tcp_name1, base_port)
    mng2 = opt_manager.add_optimizer(tcp_name2, base_port + 1)
    mng3 = opt_manager.add_optimizer(tcp_name3, base_port + 2)

    # Initialize simulator
    simulator = Simulator(model, sampling_time_sim, simulation_steps)

    # Run simulations
    with opt_manager:
        print(f"Running simulations for T = {sampler_period}s...")

        result1 = simulator.run_single_rate(mng1, x_state_0, Q, Qt, R, bounds,
                                           sampler_interval, T)

        result2 = simulator.run_single_rate(mng2, x_state_0, Q, Qt, R, bounds,
                                           sampler_interval, T)

        result3 = simulator.run_multirate(mng3, x_state_0, Q, Qt, R, bounds,
                                         sampler_interval, control_interval, N_P, T)

    return {
        "NMPC": result1,
        "Lifted NMPC": result2,
        "Lifted NMPC (multi-rate)": result3
    }


def main():
    # Simulation parameters
    sampling_time_sim = 0.001
    simulation_steps = int(10 / sampling_time_sim)
    sampling_time_control = 0.05

    # Initial conditions and cost weights
    x_state_0 = [0, np.pi, 0, 0]
    Q = [2.5, 10, 0.01, 0.01]
    Qt = [3.0, 30, 0.1, 0.02]
    R = [0.1]
    bounds = [-15, 15, -15, 15]

    # Initialize model
    model = CartPoleModel(gravity_acceleration=9.8, length=1.0,
                          mass_cart=1.0, mass_pole=0.2)

    # Run simulations for different sampler periods
    sampler_periods = [0.1, 0.25, 0.5]
    results_by_period = {}

    for period in sampler_periods:
        results_by_period[period] = run_simulation_for_period(
            period, model, sampling_time_sim, simulation_steps,
            sampling_time_control, x_state_0, Q, Qt, R, bounds
        )

    # Analyze results
    analyzer = ResultsAnalyzer()
    time = np.arange(0, sampling_time_sim * simulation_steps, sampling_time_sim)

    print("\n" + "=" * 60)
    print("Performance Metrics Summary")
    print("=" * 60)

    for period in sampler_periods:
        print(f"\n--- Sampler Period: {period}s ---")
        for label, result in results_by_period[period].items():
            state_norms = analyzer.compute_state_norms(result.get_flat_states())
            analyzer.print_metrics(f"{label} ({period})", state_norms,
                                 result.u_sequence, result.solve_time)
            print()

    # Plot results
    plotter = Plotter()
    plotter.plot_multirate_comparison(time, results_by_period)
    plotter.show()


if __name__ == "__main__":
    main()
