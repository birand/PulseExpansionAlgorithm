import numpy as np

class PulseExpansionAlgorithm:
    def __init__(self, obj_function, search_space, num_pulses=5, decay_factor=0.9, max_iterations=100, pulse_overlap_threshold=0.1, reset_threshold=10):
        self.obj_function = obj_function
        self.search_space = search_space
        self.num_pulses = num_pulses
        self.decay_factor = decay_factor
        self.max_iterations = max_iterations
        self.pulse_overlap_threshold = pulse_overlap_threshold
        self.reset_threshold = reset_threshold
        self.pulses = []
        self.initialize_pulses()

    def initialize_pulses(self):
        self.pulses = [{'center': np.random.uniform(self.search_space[0], self.search_space[1]), 
                        'radius': 1.0, 
                        'best_fitness': float('inf'), 
                        'best_position': None} for _ in range(self.num_pulses)]

    def expand_wavefront(self, pulse):
        pulse['radius'] *= self.decay_factor
        exploration_radius = pulse['radius']
        new_position = pulse['center'] + np.random.uniform(-exploration_radius, exploration_radius)
        new_position = np.clip(new_position, self.search_space[0], self.search_space[1])
        fitness = self.obj_function(new_position)
        if fitness < pulse['best_fitness']:
            pulse['best_fitness'] = fitness
            pulse['best_position'] = new_position

    def check_overlap(self, pulse1, pulse2):
        return np.abs(pulse1['center'] - pulse2['center']) < self.pulse_overlap_threshold

    def run(self):
        iteration = 0
        no_improvement_count = 0
        global_best = float('inf')
        global_best_position = None

        while iteration < self.max_iterations:
            iteration += 1
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

        return global_best_position, global_best

# Test Example - Objective function: minimizing a quadratic function
def objective_function(x):
    return (x - 3)**2  # Minimum at x = 3

# Define search space (1D for simplicity here)
search_space = [-10, 10]

# Instantiate the algorithm
pea = PulseExpansionAlgorithm(obj_function=objective_function, search_space=search_space, num_pulses=5, max_iterations=100)

# Run the algorithm
best_position, best_fitness = pea.run()
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

