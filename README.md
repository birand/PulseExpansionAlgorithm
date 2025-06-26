# Pulse Expansion Algorithm (PEA)

The **Pulse Expansion Algorithm (PEA)** is a heuristic optimization algorithm inspired by expanding waves, such as ripples in water. It provides a balance between local optimization and global exploration by dynamically expanding "pulses" (candidate solutions) across the search space. The algorithm is capable of solving a wide range of optimization problems, from simple one-dimensional functions to complex multidimensional landscapes.

## Key Features
- **Wave-based Exploration:** Pulses expand outward like ripples, exploring the search space gradually.
- **Pulse Overlap:** Areas where pulses overlap are prioritized, allowing for effective local search.
- **Decay Mechanism:** The search intensity gradually decays, preventing over-exploration of the same areas.
- **Pulse Reset:** Pulses that stagnate are reset to explore new regions, ensuring diversity in the search.

## How It Works
1. **Initialize Pulses:** Randomly select initial candidate solutions within the defined search space.
2. **Expand Wavefronts:** For each pulse, evaluate nearby solutions within an expanding radius.
3. **Check Overlap:** When two pulses overlap, prioritize the exploration of overlapping areas.
4. **Pulse Reset:** If pulses stagnate (no improvement), reset the weakest pulse to a new random location.
5. **Termination:** The algorithm stops when it converges on a solution or reaches a maximum number of iterations.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/birand/PulseExpansionAlgorithm.git
    cd PulseExpansionAlgorithm
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the Pulse Expansion Algorithm, instantiate the class with your objective function and search space, then run the algorithm to find the best solution.

### Example 1: Minimizing a Quadratic Function
```python
import numpy as np
from algorithm import PulseExpansionAlgorithm

# Define a simple quadratic objective function
def objective_function(x):
    return (x - 3)**2  # Minimum at x = 3

# Define the search space
search_space = [-10, 10]

# Instantiate the algorithm
pea = PulseExpansionAlgorithm(obj_function=objective_function, search_space=search_space, num_pulses=5, max_iterations=100)

# Run the algorithm
best_position, best_fitness = pea.run()
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
```
### Example 2: Minimizing a 2D Rosenbrock Function
```python
import numpy as np
from algorithm import PulseExpansionAlgorithm

# Define the 2D Rosenbrock function
def rosenbrock_function(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2  # Minimum at (1, 1)

# Define 2D search space
search_space = [np.array([-2, -2]), np.array([2, 2])]

# Instantiate the algorithm
pea = PulseExpansionAlgorithm(obj_function=rosenbrock_function, search_space=search_space, num_pulses=5, max_iterations=300)

# Run the algorithm
best_position, best_fitness = pea.run()
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
```
## Parameters

- **obj_function**: The objective function to minimize or maximize.
- **search_space**: The boundaries of the search space (1D or multidimensional).
- **num_pulses**: The number of pulses (candidate solutions) to explore the space.
- **decay_factor**: The rate at which the exploration radius decays.
- **max_iterations**: Maximum number of iterations before stopping.
- **pulse_overlap_threshold**: The threshold distance for pulses to overlap.
- **reset_threshold**: Number of iterations with no improvement before resetting a pulse.

# Use Cases

PEA is suitable for solving various types of optimization problems, including:

1. **Quadratic functions**: Simple convex optimization problems.
2. **Multimodal functions**: Functions with multiple local minima, where exploration is key.
3. **Multidimensional functions**: Problems involving several dimensions, such as the Rosenbrock function.
4. **Non-linear custom functions**: Tailor-made objective functions with complex landscapes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributions

Contributions are welcome! Feel free to fork this repository and submit pull requests for improvements or new features.

## Contact

For any questions or feedback, feel free to contact.