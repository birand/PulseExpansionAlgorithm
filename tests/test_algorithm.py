import pytest
from algorithm import PulseExpansionAlgorithm

# Objective function for testing
def objective_function(x):
    return (x - 3)**2  # Minimum at x = 3

@pytest.fixture
def pea_instance():
    """Provides a default instance of the PulseExpansionAlgorithm for testing."""
    return PulseExpansionAlgorithm(
        obj_function=objective_function,
        search_space=[-10, 10],
        num_pulses=10,  # Increased pulses for more robust testing
        max_iterations=200, # Increased iterations for more robust testing
        decay_factor=0.95,
        pulse_overlap_threshold=0.05,
        reset_threshold=20,
        convergence_threshold=1e-7,
        convergence_patience=20
    )

def test_finds_known_minimum(pea_instance):
    """
    Tests if the algorithm can find the known minimum of a simple quadratic function.
    """
    best_position, best_fitness = pea_instance.run()

    # We expect the algorithm to get very close to the true minimum (x=3)
    assert best_position == pytest.approx(3, abs=2e-1)
    assert best_fitness == pytest.approx(0, abs=1e-3)

def test_initialization(pea_instance):
    """
    Tests if the algorithm initializes correctly.
    """
    assert pea_instance.obj_function == objective_function
    assert pea_instance.search_space == [-10, 10]
    assert len(pea_instance.pulses) == 10

def test_pulse_initialization(pea_instance):
    """
    Tests if the pulses are initialized within the search space.
    """
    for pulse in pea_instance.pulses:
        assert -10 <= pulse['center'] <= 10
        assert pulse['radius'] == 1.0
        assert pulse['best_fitness'] == float('inf')
        assert pulse['best_position'] is None

def test_convergence(pea_instance):
    """
    Tests if the algorithm converges and stops early.
    """
    # Set a high convergence threshold to force early stopping
    pea_instance.convergence_threshold = 1.0
    pea_instance.convergence_patience = 1

    best_position, best_fitness = pea_instance.run()

    # The algorithm should stop very early, so the result will not be optimal
    assert best_position != pytest.approx(3, abs=1e-2)