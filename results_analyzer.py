import numpy as np


class ResultsAnalyzer:
    """Analyzes and computes metrics from simulation results."""

    @staticmethod
    def compute_state_norms(state_sequence_flat, dim=4):
        """Compute L2 norms of states.

        Args:
            state_sequence_flat: Flattened state sequence
            dim: State dimension

        Returns:
            List of state norms
        """
        return [
            np.linalg.norm(state_sequence_flat[i:i + dim])
            for i in range(0, len(state_sequence_flat), dim)
        ]

    @staticmethod
    def compute_rms(values):
        """Compute root mean square of values.

        Args:
            values: List or array of values

        Returns:
            RMS value
        """
        return np.sqrt(np.mean(np.square(values)))

    @staticmethod
    def compute_average(values):
        """Compute average of values.

        Args:
            values: List or array of values

        Returns:
            Average value
        """
        return np.mean(values)

    @staticmethod
    def print_metrics(label, state_norms, control_sequence, solve_times):
        """Print all metrics for a simulation result.

        Args:
            label: Description label for the results
            state_norms: State norms over time
            control_sequence: Control inputs over time
            solve_times: Solver times over time
        """
        rms_state = ResultsAnalyzer.compute_rms(state_norms)
        rms_control = ResultsAnalyzer.compute_rms(control_sequence)
        avg_solve_time = ResultsAnalyzer.compute_average(solve_times)

        print(f"RMS of state norm for {label}: {rms_state}")
        print(f"RMS of control inputs for {label}: {rms_control}")
        print(f"Average solve time for {label}: {avg_solve_time}")
