import numpy as np
from algorithm import PulseExpansionAlgorithm

# Test Example - Objective function: minimizing a quadratic function
def objective_function(x):
    """
    A simple quadratic function for testing the optimization algorithm.
    The minimum is at x = 3.
    """
    return (x - 3)**2

# Define search space (1D for simplicity here)
search_space = [-10, 10]

# Instantiate the algorithm
# You can tune the hyperparameters like num_pulses, decay_factor, etc.
pea = PulseExpansionAlgorithm(
    obj_function=objective_function,
    search_space=search_space,
    num_pulses=5,
    max_iterations=100
)

# Run the algorithm
best_position, best_fitness = pea.run()

print("--- Pulse Expansion Algorithm Example ---")
print(f"Search Space: {search_space}")
print(f"Objective Function: (x - 3)^2")
print("-" * 39)
print(f"Found Best Position: {best_position}")
print(f"Found Best Fitness: {best_fitness}")
print("-" * 39)
