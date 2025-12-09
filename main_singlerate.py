import numpy as np
from cartpole_model import CartPoleModel
from simulator import Simulator
from optimizer_manager import OptimizerManager
from results_analyzer import ResultsAnalyzer
from plotter import Plotter


def main():
    # Simulation parameters
    sampler_period = 0.05
    sampling_time_sim = 0.001
    simulation_steps = int(20 / sampling_time_sim)
    T = 20
    sampler_interval = int(sampler_period / sampling_time_sim)

    # Initial conditions and cost weights
    x_state_0 = [0, np.pi, 0, 0]
    Q = [2.5, 10, 0.01, 0.01]
    Qt = [3.0, 10, 0.02, 0.02]
    R = [0.1]
    bounds = [-2.5, 5.0, -7.5, 7.5]

    # Initialize model and simulator
    model = CartPoleModel(gravity_acceleration=9.8, length=1.0,
                          mass_cart=1.0, mass_pole=0.2)
    simulator = Simulator(model, sampling_time_sim, simulation_steps)

    # Setup optimizers
    opt_manager = OptimizerManager()
    mng1 = opt_manager.add_optimizer("OpEn/singlerate/python_cartpole/cartpole", 8333)
    mng2 = opt_manager.add_optimizer("OpEn/singlerate/python_cartpole_lifted/cartpole_lifted", 8334)

    # Run simulations
    with opt_manager:
        print("Running NMPC simulation...")
        result1 = simulator.run_single_rate(mng1, x_state_0, Q, Qt, R, bounds,
                                           sampler_interval, T)

        print("Running Lifted NMPC simulation...")
        result2 = simulator.run_single_rate(mng2, x_state_0, Q, Qt, R, bounds,
                                           sampler_interval, T)

    # Analyze results
    analyzer = ResultsAnalyzer()
    time = np.arange(0, sampling_time_sim * simulation_steps, sampling_time_sim)

    print("\n--- Performance Metrics ---")
    for label, result in [("NMPC", result1), ("Lifted NMPC", result2)]:
        state_norms = analyzer.compute_state_norms(result.get_flat_states())
        analyzer.print_metrics(label, state_norms, result.u_sequence, result.solve_time)
        print()

    # Plot results
    plotter = Plotter()
    results_dict = {
        "NMPC": result1,
        "Lifted NMPC": result2
    }
    plotter.plot_singlerate_comparison(time, results_dict, bounds)
    plotter.show()


if __name__ == "__main__":
    main()
