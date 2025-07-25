import numpy as np
import random
import mlflow
from utils import get_distance

# TSP problem definition


class TSPProblem:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

    def initial_solution(self):
        """Generate a random initial solution."""
        return random.sample(self.cities, self.num_cities)

    def energy(self, solution):
        """Calculate the total distance of the TSP solution.
        The energy is defined as the total distance of the path."""
        if not solution:
            return float('inf')
        total_distance = 0
        for i in range(len(solution)):
            city_a = solution[i]
            city_b = solution[(i + 1) % len(solution)]
            # Calculate distance between city_a and city_b
            total_distance += get_distance(city_a, city_b)
        return total_distance

    def simple_swap(self, solution):
        """ Perform a simple swap on the solution.
        This disturbance swaps two cities in the solution.
        """
        new_solution = solution[:]
        idx1, idx2 = random.sample(range(len(solution)), 2)
        new_solution[idx1], new_solution[idx2] = (
            new_solution[idx2],
            new_solution[idx1],
        )
        return new_solution

    def multiple_swap(self, solution):
        """
        Perform a multiple swap on a specif range disturbance on the solution.
        """
        new_solution = solution[:]
        indices = random.sample(range(len(solution)), 2)
        idx1, idx2 = indices[0], indices[1]
        new_solution[idx1:idx2] = reversed(new_solution[idx1:idx2])
        return new_solution

    def disturbance(self, type, solution):
        """Apply a disturbance to the solution based on the type."""
        if type == "simple_swap":
            return self.simple_swap(solution)
        elif type == "multiple_swap":
            return self.multiple_swap(solution)
        else:
            raise ValueError("Unknown disturbance type")


class SimulatedAnnealing:
    def __init__(self, initial_temperature, cooling_rate):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(self, problem):
        current_solution = problem.initial_solution()
        current_energy = problem.energy(current_solution)
        temperature = self.initial_temperature

        while temperature > 1:
            new_solution = problem.get_neighbor(current_solution)
            new_energy = problem.energy(new_solution)

            if (
                self.acceptance_probability(
                    current_energy, new_energy, temperature
                )
                > np.random.uniform()
            ):
                current_solution = new_solution
                current_energy = new_energy

            temperature *= self.cooling_rate

        return current_solution

    def acceptance_probability(self, current_energy, new_energy, temperature):
        if new_energy < current_energy:
            return 1.0
        return np.exp((current_energy - new_energy) / temperature)
