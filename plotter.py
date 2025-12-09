import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)


class Plotter:
    """Handles all plotting functionality for simulation results."""

    def __init__(self, font_family='Times New Roman'):
        plt.rcParams['font.family'] = font_family

    @staticmethod
    def _setup_subplot(ylabel, xlabel=None, show_legend=True, legend_loc='upper right',
                       xlim_right=None, grid=True):
        """Common subplot configuration."""
        plt.ylabel(ylabel)
        if xlabel:
            plt.xlabel(xlabel)
        if grid:
            plt.grid(True)
        if show_legend:
            plt.legend(loc=legend_loc, bbox_to_anchor=(1, 0.9))
        if xlim_right is not None:
            plt.xlim(left=0, right=xlim_right)

    def plot_singlerate_comparison(self, time, results_dict, bounds=None):
        """Plot comparison of single-rate simulation results.

        Args:
            time: Time vector
            results_dict: Dict with keys as labels and values as SimulationResult objects
            bounds: Optional bounds [x_min, x_max, x_dot_min, x_dot_max]
        """
        right_limit = time[-1]
        labels = list(results_dict.keys())
        line_styles = ['b-', 'r--', 'g-', 'c:', 'm-.']

        # Plot x1 (cart position)
        plt.figure()
        for i, (label, result) in enumerate(results_dict.items()):
            state_flat = result.get_flat_states()
            plt.plot(time, state_flat[0::4], line_styles[i % len(line_styles)], label=label)

        if bounds:
            plt.axhline(bounds[0], color='k', linestyle='-.', label="Bounds")
            plt.axhline(bounds[1], color='k', linestyle='-.')

        self._setup_subplot(r'$x_1$ (m)', 'Time (s)', xlim_right=right_limit)

        # Plot x3 (cart velocity)
        plt.figure()
        for i, (label, result) in enumerate(results_dict.items()):
            state_flat = result.get_flat_states()
            plt.plot(time, state_flat[2::4], line_styles[i % len(line_styles)], label=label)

        if bounds:
            plt.axhline(bounds[2], color='k', linestyle='-.', label="Bounds")
            plt.axhline(bounds[3], color='k', linestyle='-.')

        self._setup_subplot(r'$x_3$ (m/s)', 'Time (s)', xlim_right=right_limit)

        # Plot control inputs
        plt.figure()
        for i, (label, result) in enumerate(results_dict.items()):
            plt.plot(time, result.u_sequence, line_styles[i % len(line_styles)], label=label)

        self._setup_subplot(r'Control Input: $u$', 'Time (s)', xlim_right=right_limit)
        plt.tight_layout()

        # Plot state norms
        plt.figure()
        for i, (label, result) in enumerate(results_dict.items()):
            state_flat = result.get_flat_states()
            state_norms = [np.linalg.norm(state_flat[j:j+4]) for j in range(0, len(state_flat), 4)]
            plt.plot(time, state_norms, line_styles[i % len(line_styles)], label=label)

        self._setup_subplot(r'$||x||_{2}$', 'Time (s)', xlim_right=right_limit)
        plt.tight_layout()

        # Plot solve times
        plt.figure()
        for i, (label, result) in enumerate(results_dict.items()):
            plt.plot(time, result.solve_time, line_styles[i % len(line_styles)], label=label)

        self._setup_subplot('Solve Time (ms)', 'Time (s)', xlim_right=right_limit)
        plt.tight_layout()

    def plot_multirate_comparison(self, time, results_by_period):
        """Plot comparison of multirate simulation results.

        Args:
            time: Time vector
            results_by_period: Dict with sampler periods as keys, each containing
                             dict of {label: SimulationResult}
        """
        right_limit = time[-1]
        line_styles = {'NMPC': 'b-', 'Lifted NMPC': 'r--', 'Lifted NMPC (multi-rate)': 'g-'}

        # Plot control inputs
        plt.figure()
        periods = sorted(results_by_period.keys())
        for idx, period in enumerate(periods, 1):
            plt.subplot(3, 1, idx)
            for label, result in results_by_period[period].items():
                plt.plot(time, result.u_sequence, line_styles.get(label, 'k-'), label=label)

            ylabel = f'T = {period} (s)'
            show_legend = (idx == 1)
            legend_loc = 'lower right' if show_legend else None
            self._setup_subplot(ylabel, xlabel='Time (s)' if idx == 3 else None,
                              show_legend=show_legend, legend_loc=legend_loc,
                              xlim_right=right_limit)

        plt.tight_layout()

        # Plot cart position
        plt.figure()
        for idx, period in enumerate(periods, 1):
            plt.subplot(3, 1, idx)
            for label, result in results_by_period[period].items():
                state_flat = result.get_flat_states()
                plt.plot(time, [state_flat[i] for i in range(0, len(state_flat), 4)],
                        line_styles.get(label, 'k-'), label=label)

            ylabel = f'T = {period} (s)'
            show_legend = (idx == 1)
            legend_loc = 'lower right' if show_legend else None
            self._setup_subplot(ylabel, xlabel='Time (s)' if idx == 3 else None,
                              show_legend=show_legend, legend_loc=legend_loc,
                              xlim_right=right_limit)

        plt.tight_layout()

        # Plot pole angle
        plt.figure()
        for idx, period in enumerate(periods, 1):
            plt.subplot(3, 1, idx)
            for label, result in results_by_period[period].items():
                state_flat = result.get_flat_states()
                plt.plot(time, [state_flat[i] for i in range(1, len(state_flat), 4)],
                        line_styles.get(label, 'k-'), label=label)

            ylabel = f'T = {period} (s)'
            show_legend = (idx == 1)
            legend_loc = 'lower right' if show_legend else None
            self._setup_subplot(ylabel, xlabel='Time (s)' if idx == 3 else None,
                              show_legend=show_legend, legend_loc=legend_loc,
                              xlim_right=right_limit)

        plt.tight_layout()

        # Plot state norms
        plt.figure()
        for idx, period in enumerate(periods, 1):
            plt.subplot(3, 1, idx)
            for label, result in results_by_period[period].items():
                state_flat = result.get_flat_states()
                state_norms = [np.linalg.norm(state_flat[j:j+4]) for j in range(0, len(state_flat), 4)]
                plt.plot(time, state_norms, line_styles.get(label, 'k-'), label=label)

            ylabel = f'T = {period} (s)'
            show_legend = (idx == 1)
            legend_loc = 'upper right' if show_legend else None
            self._setup_subplot(ylabel, xlabel='Time (s)' if idx == 3 else None,
                              show_legend=show_legend, legend_loc=legend_loc,
                              xlim_right=right_limit)

        plt.tight_layout()

        # Plot solve times
        plt.figure()
        for idx, period in enumerate(periods, 1):
            plt.subplot(3, 1, idx)
            for label, result in results_by_period[period].items():
                capped_times = np.minimum(result.solve_time, 45)
                plt.plot(time, capped_times, line_styles.get(label, 'k-'), label=label)

            ylabel = f'T = {period} (s)'
            show_legend = (idx == 1)
            legend_loc = 'upper right' if show_legend else None
            self._setup_subplot(ylabel, xlabel='Time (s)' if idx == 3 else None,
                              show_legend=show_legend, legend_loc=legend_loc,
                              xlim_right=right_limit)

        plt.tight_layout()

    @staticmethod
    def show():
        """Display all plots."""
        plt.show()
