import numpy as np
from algorithm import PulseExpansionAlgorithm

# Test Example 1: Minimizing a quadratic function (1D)
def objective_function_1d(x):
    """
    A simple quadratic function for testing the optimization algorithm.
    The minimum is at x = 3.
    """
    return (x - 3)**2

# Define search space (1D)
search_space_1d = [-10, 10]

# Instantiate the algorithm for 1D
pea_1d = PulseExpansionAlgorithm(
    obj_function=objective_function_1d,
    search_space=search_space_1d,
    num_pulses=5,
    max_iterations=100
)

# Run the algorithm for 1D
best_position_1d, best_fitness_1d = pea_1d.run()

print("--- Pulse Expansion Algorithm Example (1D Quadratic) ---")
print(f"Search Space: {search_space_1d}")
print(f"Objective Function: (x - 3)^2")
print("-" * 50)
print(f"Found Best Position: {best_position_1d}")
print(f"Found Best Fitness: {best_fitness_1d}")
print("-" * 50)

# Test Example 2: Minimizing a 2D Rosenbrock function
def rosenbrock_function_2d(x):
    """
    The Rosenbrock function is a non-convex function used as a performance test problem
    for optimization algorithms. The global minimum is at (1, 1) with a value of 0.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Define search space (2D)
search_space_2d = [[-2, 2], [-2, 2]]

# Instantiate the algorithm for 2D
pea_2d = PulseExpansionAlgorithm(
    obj_function=rosenbrock_function_2d,
    search_space=search_space_2d,
    num_pulses=10,
    max_iterations=500,
    decay_factor=0.98,
    pulse_overlap_threshold=0.1,
    reset_threshold=30
)

# Run the algorithm for 2D
best_position_2d, best_fitness_2d = pea_2d.run()

print("\n--- Pulse Expansion Algorithm Example (2D Rosenbrock) ---")
print(f"Search Space: {search_space_2d}")
print(f"Objective Function: Rosenbrock (2D)")
print("-" * 50)
print(f"Found Best Position: {best_position_2d}")
print(f"Found Best Fitness: {best_fitness_2d}")
print("-" * 50)