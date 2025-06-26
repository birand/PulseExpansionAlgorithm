import numpy as np

class PulseExpansionAlgorithm:
    """
    A custom optimization algorithm that uses a set of "pulses" to explore a search space
    and find the minimum value of a given objective function.

    The algorithm works by initializing a number of pulses within the search space.
    Each pulse expands its wavefront to explore its local neighborhood.
    The algorithm includes mechanisms for handling overlapping pulses and for resetting
    pulses that are not showing improvement, to avoid getting stuck in local minima.
    """
    def __init__(self, obj_function, search_space, num_pulses=5, decay_factor=0.9, max_iterations=100, pulse_overlap_threshold=0.1, reset_threshold=10, convergence_threshold=1e-6, convergence_patience=10):
        """
        Initializes the Pulse Expansion Algorithm.

        Args:
            obj_function (callable): The objective function to minimize.
            search_space (list or tuple): A list or tuple defining the search space, e.g., [-10, 10].
            num_pulses (int): The number of pulses to use for exploration.
            decay_factor (float): The factor by which the pulse radius decays in each iteration.
            max_iterations (int): The maximum number of iterations to run the algorithm.
            pulse_overlap_threshold (float): The threshold for considering two pulses as overlapping.
            reset_threshold (int): The number of iterations without improvement after which to reset a weak pulse.
            convergence_threshold (float): The threshold for fitness improvement to consider the algorithm converged.
            convergence_patience (int): The number of iterations to wait for improvement before stopping.
        """
        self.obj_function = obj_function
        self.search_space = search_space
        self.num_pulses = num_pulses
        self.decay_factor = decay_factor
        self.max_iterations = max_iterations
        self.pulse_overlap_threshold = pulse_overlap_threshold
        self.reset_threshold = reset_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.pulses = []
        self.initialize_pulses()

    def initialize_pulses(self):
        """
        Initializes the pulses at random positions within the search space.
        """
        self.pulses = [{'center': np.random.uniform(self.search_space[0], self.search_space[1]), 
                        'radius': 1.0, 
                        'best_fitness': float('inf'), 
                        'best_position': None} for _ in range(self.num_pulses)]

    def expand_wavefront(self, pulse):
        """
        Expands the wavefront of a single pulse to explore the search space.

        Args:
            pulse (dict): The pulse to expand.
        """
        pulse['radius'] *= self.decay_factor
        exploration_radius = pulse['radius']
        new_position = pulse['center'] + np.random.uniform(-exploration_radius, exploration_radius)
        new_position = np.clip(new_position, self.search_space[0], self.search_space[1])
        fitness = self.obj_function(new_position)
        if fitness < pulse['best_fitness']:
            pulse['best_fitness'] = fitness
            pulse['best_position'] = new_position

    def check_overlap(self, pulse1, pulse2):
        """
        Checks if two pulses are overlapping based on their center positions.

        Args:
            pulse1 (dict): The first pulse.
            pulse2 (dict): The second pulse.

        Returns:
            bool: True if the pulses are overlapping, False otherwise.
        """
        return np.abs(pulse1['center'] - pulse2['center']) < self.pulse_overlap_threshold

    def run(self):
        """
        Runs the main loop of the Pulse Expansion Algorithm.

        Returns:
            tuple: A tuple containing the best position found and its corresponding fitness value.
        """
        iteration = 0
        no_improvement_count = 0
        convergence_counter = 0
        global_best = float('inf')
        global_best_position = None

        while iteration < self.max_iterations:
            iteration += 1
            previous_global_best = global_best

            for i, pulse in enumerate(self.pulses):
                self.expand_wavefront(pulse)
                
                # Check if overlap exists
                for j, other_pulse in enumerate(self.pulses):
                    if i != j and self.check_overlap(pulse, other_pulse):
                        self.expand_wavefront(pulse)  # Prioritize overlap area
                
                # Global best check
                if pulse['best_fitness'] < global_best:
                    global_best = pulse['best_fitness']
                    global_best_position = pulse['best_position']
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Reset weakest pulse after no improvement
            if no_improvement_count > self.reset_threshold:
                weakest_pulse = max(self.pulses, key=lambda p: p['best_fitness'])
                weakest_pulse['center'] = np.random.uniform(self.search_space[0], self.search_space[1])
                weakest_pulse['radius'] = 1.0
                weakest_pulse['best_fitness'] = float('inf')
                weakest_pulse['best_position'] = None
                no_improvement_count = 0

            # Check for convergence
            if abs(previous_global_best - global_best) < self.convergence_threshold:
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter >= self.convergence_patience:
                print(f"Convergence reached at iteration {iteration}. Stopping.")
                break

        return global_best_position, global_best