import numpy as np
import random
import mlflow
import geopandas as gpd
from geobr import read_municipality
import matplotlib.pyplot as plt
from .utils import get_distance
import logging
import tempfile
import os

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TSPProblem:
    """Class representing the Traveling Salesman Problem (TSP) for a set of cities.
    Attributes:
        cities (GeoDataFrame): GeoDataFrame containing city information.
        num_cities (int): Number of cities in the problem.
        initial_solution (GeoDataFrame): Initial solution for the TSP.
    """

    def __init__(self, cities_gdf: gpd.GeoDataFrame, initial_solution=None):
        self.cities = cities_gdf
        self.num_cities = len(cities_gdf)
        # Pre-compute distance matrix for performance
        self.distance_matrix = self._compute_distance_matrix()
        # Create a mapping from coordinates to original indices for efficient lookup
        self._coord_to_index = {}
        for idx, row in cities_gdf.iterrows():
            coord_key = (row["latitude"], row["longitude"])
            self._coord_to_index[coord_key] = idx

        if initial_solution is None:
            self.initial_solution = self._initial_solution()
        else:
            self.initial_solution = initial_solution

    def _initial_solution(
        self,
    ):
        """Generate a random initial solution."""
        return self.cities.sample(
            n=self.num_cities, random_state=42
        ).reset_index(drop=True)

    def _compute_distance_matrix(self):
        """Pre-compute all pairwise distances between cities for performance."""
        n = len(self.cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                coords_a = (
                    self.cities.iloc[i]["latitude"],
                    self.cities.iloc[i]["longitude"],
                )
                coords_b = (
                    self.cities.iloc[j]["latitude"],
                    self.cities.iloc[j]["longitude"],
                )
                dist = get_distance(coords_a, coords_b)
                distances[i][j] = distances[j][i] = dist
        return distances

    def energy(self, solution):
        """Calculate the total distance of the TSP solution using pre-computed distance matrix."""
        if solution is None or len(solution) == 0:
            return float("inf")

        total_distance = 0.0
        # Get the original city indices in the order they appear in the solution
        solution_indices = []
        for i in range(len(solution)):
            city_row = solution.iloc[i]
            coord_key = (city_row["latitude"], city_row["longitude"])
            original_idx = self._coord_to_index[coord_key]
            solution_indices.append(original_idx)

        for i in range(len(solution_indices)):
            idx_a = solution_indices[i]
            idx_b = solution_indices[(i + 1) % len(solution_indices)]
            total_distance += float(self.distance_matrix[idx_a][idx_b])
        return total_distance

    def simple_swap(self, solution):
        """Optimized simple swap with reduced DataFrame operations."""
        new_solution = solution.copy()
        n = len(solution)
        idx1, idx2 = random.sample(range(n), 2)
        # Use more efficient iloc swapping
        temp = new_solution.iloc[idx1].copy()
        new_solution.iloc[idx1] = new_solution.iloc[idx2]
        new_solution.iloc[idx2] = temp
        return new_solution

    def multiple_swap(self, solution):
        """Optimized multiple swap with reduced operations."""
        new_solution = solution.copy()
        n = len(solution)
        indices = sorted(random.sample(range(n), 2))
        idx1, idx2 = indices[0], indices[1]
        # Reverse the slice more efficiently
        segment = new_solution.iloc[idx1:idx2].iloc[::-1]
        new_solution.iloc[idx1:idx2] = segment.values
        return new_solution

    def disturbance(self, type, solution):
        if type == "simple_swap":
            return self.simple_swap(solution)
        elif type == "multiple_swap":
            return self.multiple_swap(solution)
        else:
            raise ValueError("Unknown disturbance type")

    def plot_solution(
        self,
        city_order,
        gdf,
        brazil,
        ax=None,
        title="TSP Solution for Brazilian Cities",
        show_names_only=False,
        show_plot=True,
    ):
        """
        Plots the TSP solution on the Brazil map.

        Parameters:
            city_order (list): List of indices or city names in the visiting order.
            gdf (GeoDataFrame): GeoDataFrame with city info (must include 'longitude' and 'latitude').
            brazil (GeoDataFrame): GeoDataFrame with Brazil state boundaries.
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new one.
            title (str): Title for the plot.
            show_names_only (bool): If True, only show city names in visiting order.
            show_plot (bool): If True, displays the plot with plt.show().
        """

        # Prepare coordinates in visiting order
        if isinstance(city_order[0], str):
            visited_gdf = gdf[gdf["nome"].isin(city_order)]
            coords = (
                visited_gdf.set_index("nome")
                .loc[city_order][["longitude", "latitude"]]
                .values
            )
            names = city_order
        else:
            visited_gdf = gdf.iloc[city_order]
            coords = visited_gdf[["longitude", "latitude"]].values
            names = visited_gdf["nome"].values

        # Optionally close the loop
        coords = np.vstack([coords, coords[0]])
        names = list(names) + [names[0]]

        if show_names_only:
            print("Visited cities in order:")
            for name in names[:-1]:
                print(name)
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        brazil.plot(ax=ax, color="white", edgecolor="black")
        visited_gdf.plot(ax=ax, color="red", markersize=30, zorder=2)

        # Plot the TSP path
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color="black",
            linewidth=2,
            zorder=3,
            label="TSP Path",
        )

        # Annotate cities
        for idx, (lon, lat) in enumerate(coords[:-1]):
            ax.text(lon, lat, names[idx], fontsize=9, ha="right", va="bottom")

        ax.set_title(title)
        ax.legend()
        if show_plot:
            plt.show()

    def log_solution_to_mlflow(
        self, solution, solution_name="tsp_solution", brazil_gdf=None
    ):
        """
        Log the TSP solution details to MLflow.

        Parameters:
            solution: The solution (GeoDataFrame) to log
            solution_name: Name for the logged solution
            brazil_gdf: GeoDataFrame for Brazil boundaries (optional, for plotting)
        """
        if solution is not None and len(solution) > 0:
            # Log solution energy/distance
            energy = self.energy(solution)
            # mlflow.log_metric(f"{solution_name}_energy", energy)

            # Log city order as a parameter (truncated if too long)
            city_names = (
                solution["nome"].tolist()
                if "nome" in solution.columns
                else [f"city_{i}" for i in range(len(solution))]
            )
            city_order_str = " -> ".join(city_names)

            # MLflow has a parameter value length limit, so truncate if necessary
            if len(city_order_str) > 500:
                city_order_str = city_order_str[:497] + "..."

            mlflow.log_param(f"{solution_name}_route", city_order_str)

            # Log plot of the solution if brazil_gdf is provided
            if brazil_gdf is not None:

                fig, ax = plt.subplots(figsize=(12, 12))
                self.plot_solution(
                    city_order=city_names,
                    gdf=solution,
                    brazil=brazil_gdf,
                    ax=ax,
                    title=f"TSP Solution: {solution_name}",
                    show_plot=False,
                )
                # Save plot to a temporary file without displaying it
                with tempfile.TemporaryDirectory() as tmpdir:
                    plot_path = os.path.join(
                        tmpdir, f"{solution_name}_plot.png"
                    )
                    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    mlflow.log_artifact(plot_path, artifact_path="plots")


class SimulatedAnnealing:

    def __init__(
        self,
        initial_temperature,
        total_temperatures=10,
        method="logarithmic",
        total_iterations=1000,
        disturbance_type="simple_swap",
        keep_best=True,
        alpha=0.95,
        # Performance optimization parameters
        log_interval=10,  # Log metrics every N iterations instead of every iteration
        console_log_interval=500,  # Console logging frequency
        plot_interval=5,  # Generate plots every N temperature steps
        performance_mode=False,  # When True, minimal logging for speed
        # experiment_name="TSP_Simulated_Annealing",
    ):
        """
        Initializes the Simulated Annealing algorithm.

        Parameters:
            experiment_name (str): Name for the MLflow experiment
            log_interval (int): Log metrics every N iterations (default: 10)
            console_log_interval (int): Console logging frequency (default: 500)
            plot_interval (int): Generate plots every N temperature steps (default: 5)
            performance_mode (bool): When True, uses minimal logging for maximum speed
        """
        self.initial_temperature = initial_temperature
        self.total_iterations = total_iterations
        self.k = 0
        self.total_temperatures = total_temperatures
        self.method = method
        self.disturbance_type = disturbance_type
        self.keep_best = keep_best
        self.alpha = alpha

        # Performance optimization settings
        if performance_mode:
            self.log_interval = 100
            self.console_log_interval = 1000
            self.plot_interval = float("inf")  # No intermediate plots
        else:
            self.log_interval = log_interval
            self.console_log_interval = console_log_interval
            self.plot_interval = plot_interval

        # self.experiment_name = experiment_name

        if method not in ["cauchy", "logarithmic", "exponential"]:
            raise ValueError(
                "Method must be either 'cauchy', 'logarithmic' or 'exponential'"
            )

        # Set MLflow experiment
        # mlflow.set_experiment(self.experiment_name)

    def optimize(self, problem: TSPProblem, brazil_gdf=None):
        current_solution = problem.initial_solution
        current_energy = problem.energy(current_solution)
        minimum_energy = current_energy
        minimum_solution = current_solution.copy()

        # Store metrics for logging after completion
        metrics_history = {
            "temperatures": [],
            "current_energies": [],
            "min_energies": [],
            "best_energies": [],
            # "final_solution_energies": [],
            # "best_solution_energies": [],
            "solutions_history": [],
        }

        total_iterations_count = 0
        accepted_moves = 0
        end = False

        while not (end):
            if self.method == "cauchy":
                temperature = self.initial_temperature / (1 + self.k)
            elif self.method == "logarithmic":
                temperature = self.initial_temperature / np.log2(2 + self.k)
            elif self.method == "exponential":
                temperature = self.initial_temperature * (self.alpha**self.k)
            else:
                temperature = self.initial_temperature  # fallback

            # Store temperature for later logging
            metrics_history["temperatures"].append((temperature, self.k))
            LOGGER.info(f"Temperature at step {self.k}: {temperature}")

            for n in range(0, self.total_iterations):
                new_solution = problem.disturbance(
                    self.disturbance_type, current_solution
                )
                new_energy = problem.energy(new_solution)

                acceptance_prob = self.acceptance_probability(
                    current_energy, new_energy, temperature
                )

                if acceptance_prob > np.random.uniform():
                    current_solution = new_solution
                    current_energy = new_energy
                    accepted_moves += 1

                    if current_energy < minimum_energy:
                        minimum_energy = current_energy
                        minimum_solution = current_solution.copy()
                        # Store best energy for later logging
                        metrics_history["best_energies"].append(
                            (minimum_energy, total_iterations_count)
                        )

                # Store metrics for later logging
                metrics_history["current_energies"].append(
                    (current_energy, total_iterations_count)
                )
                metrics_history["min_energies"].append(
                    (minimum_energy, total_iterations_count)
                )

                # Optimized console logging with reduced frequency
                if np.remainder(n + 1, self.console_log_interval) == 0:
                    LOGGER.info(
                        f"Step: {self.k}, Temp: {temperature:.2f}, Iter: {n + 1}, "
                        f"Min: {minimum_energy:.2f}, Current: {current_energy:.2f}"
                    )

                total_iterations_count += 1

            # Store solutions at the end of each temperature k for later logging
            final_solution_k_energy = problem.energy(current_solution)
            best_solution_k_energy = problem.energy(minimum_solution)

            # metrics_history["final_solution_energies"].append(
            #     (f"final_solution_k_{self.k}_energy", final_solution_k_energy)
            # )
            # metrics_history["best_solution_energies"].append(
            #     (f"best_solution_k_{self.k}_energy", best_solution_k_energy)
            # )

            # Store solutions for later logging
            should_plot = (self.k % self.plot_interval == 0) or (
                self.k == self.total_temperatures - 1
            )
            metrics_history["solutions_history"].append(
                {
                    "current_solution": current_solution.copy(),
                    "minimum_solution": minimum_solution.copy(),
                    "step": self.k,
                    "should_plot": should_plot,
                }
            )

            self.k += 1
            # Start with the best solution from the previous temperature
            if self.keep_best:
                current_solution = minimum_solution.copy()
            if self.k >= self.total_temperatures:
                end = True

        # ========== ALL MLFLOW LOGGING HAPPENS HERE AFTER OPTIMIZATION ==========

        # Log hyperparameters
        mlflow.log_param("initial_temperature", self.initial_temperature)
        mlflow.log_param("total_iterations", self.total_iterations)
        mlflow.log_param("total_temperatures", self.total_temperatures)
        mlflow.log_param("method", self.method)
        mlflow.log_param("disturbance_type", self.disturbance_type)
        mlflow.log_param("num_cities", problem.num_cities)

        # Log initial solution energy
        initial_energy = problem.energy(problem.initial_solution)
        mlflow.log_metric("initial_energy", initial_energy)

        # Log initial solution details with plot
        problem.log_solution_to_mlflow(
            problem.initial_solution, "initial_solution", brazil_gdf
        )

        # Log temperature history
        for temperature, step in metrics_history["temperatures"]:
            mlflow.log_metric("temperature", temperature, step=step)

        # Log current energy history (sample every log_interval to avoid too many points)
        for i, (energy, step) in enumerate(
            metrics_history["current_energies"]
        ):
            if i % self.log_interval == 0:  # Sample based on log_interval
                mlflow.log_metric("current_energy", energy, step=step)

        # Log minimum energy history (sample every log_interval)
        for i, (energy, step) in enumerate(metrics_history["min_energies"]):
            if i % self.log_interval == 0:  # Sample based on log_interval
                mlflow.log_metric("min_energy_so_far", energy, step=step)

        # Log all best energy improvements
        for energy, step in metrics_history["best_energies"]:
            mlflow.log_metric("best_energy", energy, step=step)

        # # Log final and best solution energies for each temperature step
        # for metric_name, energy in metrics_history["final_solution_energies"]:
        #     mlflow.log_metric(metric_name, energy)

        # for metric_name, energy in metrics_history["best_solution_energies"]:
        #     mlflow.log_metric(metric_name, energy)

        # Log solutions history
        for solution_data in metrics_history["solutions_history"]:
            step = solution_data["step"]
            should_plot = solution_data["should_plot"]

            problem.log_solution_to_mlflow(
                solution_data["current_solution"],
                f"final_solution_k_{step}",
                brazil_gdf if should_plot else None,
            )
            problem.log_solution_to_mlflow(
                solution_data["minimum_solution"],
                f"best_solution_k_{step}",
                brazil_gdf if should_plot else None,
            )

        # Log final metrics
        mlflow.log_metric("final_energy", minimum_energy)
        mlflow.log_metric("total_iterations_executed", total_iterations_count)
        mlflow.log_metric("accepted_moves", accepted_moves)
        mlflow.log_metric(
            "acceptance_rate", accepted_moves / total_iterations_count
        )

        # Log improvement ratio
        improvement_ratio = (
            (initial_energy - minimum_energy) / initial_energy
            if initial_energy > 0
            else 0
        )
        mlflow.log_metric("improvement_ratio", improvement_ratio)

        # Set tags for easy filtering
        mlflow.set_tag("algorithm", "simulated_annealing")
        mlflow.set_tag("problem_type", "tsp")
        mlflow.set_tag("cooling_method", self.method)

        # Log final solution details with plot
        problem.log_solution_to_mlflow(
            minimum_solution, "final_solution", brazil_gdf
        )

        return minimum_solution

    def acceptance_probability(self, current_energy, new_energy, temperature):
        return np.exp((current_energy - new_energy) / temperature)
