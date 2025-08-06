import os
import sys
import logging
import logging.config
import yaml
import itertools
import mlflow
from geobr import read_state
from .models import TSPProblem, SimulatedAnnealing
from .load import load_cities

import geodatasets
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def gridsearch_hyperparams(
    param_grid, adaptative=False, previous_results=None
):
    """
    Generate hyperparameter combinations.
    If adaptative=True, narrows the search space based on previous_results.
    previous_results: list of dicts with 'params' and 'score'.
    """
    if not adaptative or not previous_results:
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
    else:
        # Example: select top N results and refine grid around them
        top_n = 3
        sorted_results = sorted(previous_results, key=lambda x: x["score"])
        top_params = [r["params"] for r in sorted_results[:top_n]]
        for base in top_params:
            refined_grid = {}
            for k, vlist in param_grid.items():
                base_val = base[k]
                # Find neighbors in vlist
                idx = vlist.index(base_val)
                neighbors = vlist[max(0, idx - 1) : min(len(vlist), idx + 2)]
                refined_grid[k] = neighbors
            for instance in itertools.product(*refined_grid.values()):
                yield dict(zip(refined_grid.keys(), instance))


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    config = load_config(config_path)

    param_grid = config["hyperparams"]
    sa_config = config.get("sa_config", {})
    mlflow.set_experiment(
        config.get("mlflow_experiment", "TSP_Simulated_Annealing")
    )

    # Configuration for adaptive search
    use_adaptive = config.get("adaptive_search", False)
    max_iterations = config.get("max_search_iterations", 2)

    # Load Brazil state boundaries
    brazil = read_state()
    LOGGER.info(f"Loaded Brazil state boundaries: {brazil.shape[0]} states")

    # Load cities from geobr (e.g., select cities from RJ)
    cities_gdf = load_cities(
        config.get("cities_data_path", "data/cities.csv"),
        endpoint=config.get(
            "ibge_endpoint",
            "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2024/variaveis/9324?localidades=N6[all]",
        ),
    )

    n_cities = config.get("n_cities", 10)
    if n_cities > len(cities_gdf):
        raise ValueError(
            f"Requested {n_cities} cities, but only {len(cities_gdf)} available."
        )

    cities_gdf = cities_gdf.sample(n=n_cities, random_state=42).reset_index(
        drop=True
    )

    problem = TSPProblem(cities_gdf)

    # Store results for adaptive search
    all_results = []

    if use_adaptive:
        LOGGER.info(
            f"Running adaptive grid search with {max_iterations} iterations"
        )

        for iteration in range(max_iterations):
            LOGGER.info(
                f"--- Adaptive Search Iteration {iteration + 1}/{max_iterations} ---"
            )

            # First iteration uses full grid, subsequent use adaptive
            use_adaptive_this_iter = iteration > 0
            previous_results = all_results if use_adaptive_this_iter else None

            iteration_results = []

            for params in gridsearch_hyperparams(
                param_grid, use_adaptive_this_iter, previous_results
            ):
                LOGGER.info(f"Running with hyperparameters: {params}")
                with mlflow.start_run():
                    # Log iteration info
                    mlflow.log_param("search_iteration", iteration + 1)
                    mlflow.log_param("is_adaptive", use_adaptive_this_iter)
                    mlflow.log_params(params)
                    mlflow.log_params(sa_config)

                    sa = SimulatedAnnealing(**params, **sa_config)
                    best_solution = sa.optimize(problem, brazil_gdf=brazil)
                    score = problem.energy(best_solution)
                    mlflow.log_metric("score", score)

                    # Get run ID safely
                    current_run = mlflow.active_run()
                    run_id = (
                        current_run.info.run_id if current_run else "unknown"
                    )

                    LOGGER.info(f"Run {run_id} finished with score: {score}")

                    # Store result for adaptive search
                    result = {
                        "params": params,
                        "score": score,
                        "run_id": run_id,
                    }
                    iteration_results.append(result)
                    all_results.append(result)

            # Log best result of this iteration
            if iteration_results:
                best_iter = min(iteration_results, key=lambda x: x["score"])
                LOGGER.info(
                    f"Best result in iteration {iteration + 1}: score={best_iter['score']:.2f}, params={best_iter['params']}"
                )
    else:
        LOGGER.info("Running standard grid search")
        for params in gridsearch_hyperparams(param_grid):
            LOGGER.info(f"Running with hyperparameters: {params}")
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_params(sa_config)
                sa = SimulatedAnnealing(**params, **sa_config)
                best_solution = sa.optimize(problem, brazil_gdf=brazil)
                score = problem.energy(best_solution)
                mlflow.log_metric("score", score)

                # Get run ID safely
                current_run = mlflow.active_run()
                run_id = current_run.info.run_id if current_run else "unknown"

                LOGGER.info(f"Run {run_id} finished with score: {score}")

    # Log overall best result
    if all_results:
        overall_best = min(all_results, key=lambda x: x["score"])
        LOGGER.info(
            f"Overall best result: score={overall_best['score']:.2f}, params={overall_best['params']}, run_id={overall_best['run_id']}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tsp <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
