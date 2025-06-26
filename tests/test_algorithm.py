import pytest
import numpy as np
from algorithm import PulseExpansionAlgorithm

# Objective function for 1D testing
def objective_function_1d(x):
    return (x - 3)**2  # Minimum at x = 3

# Objective function for 2D testing (Rosenbrock)
def rosenbrock_function_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2  # Minimum at (1, 1)

@pytest.fixture
def pea_instance_1d():
    """Provides a default instance of the PulseExpansionAlgorithm for 1D testing."""
    return PulseExpansionAlgorithm(
        obj_function=objective_function_1d,
        search_space=[-10, 10],
        num_pulses=20,
        max_iterations=500,
        decay_factor=0.90,
        pulse_overlap_threshold=0.05,
        reset_threshold=25,
        convergence_threshold=1e-7,
        convergence_patience=25
    )

@pytest.fixture
def pea_instance_2d():
    """Provides an instance of the PulseExpansionAlgorithm for 2D testing."""
    return PulseExpansionAlgorithm(
        obj_function=rosenbrock_function_2d,
        search_space=[[-2, 2], [-2, 2]],
        num_pulses=40,
        max_iterations=2000,
        decay_factor=0.92,
        pulse_overlap_threshold=0.05,
        reset_threshold=50,
        convergence_threshold=1e-6,
        convergence_patience=40
    )

def test_finds_known_minimum_1d(pea_instance_1d):
    """
    Tests if the algorithm can find the known minimum of a simple quadratic function (1D).
    """
    best_position, best_fitness = pea_instance_1d.run()

    assert best_position == pytest.approx(3, abs=3e-1)
    assert best_fitness == pytest.approx(0, abs=2e-3)

def test_initialization_1d(pea_instance_1d):
    """
    Tests if the 1D algorithm initializes correctly.
    """
    assert pea_instance_1d.obj_function == objective_function_1d
    assert pea_instance_1d.search_space == [-10, 10]
    assert len(pea_instance_1d.pulses) == 20

def test_pulse_initialization_1d(pea_instance_1d):
    """
    Tests if the 1D pulses are initialized within the search space.
    """
    for pulse in pea_instance_1d.pulses:
        assert -10 <= pulse['center'] <= 10
        assert pulse['radius'] == 1.0
        assert pulse['best_fitness'] == float('inf')
        assert pulse['best_position'] is None

def test_convergence_1d(pea_instance_1d):
    """
    Tests if the 1D algorithm converges and stops early.
    """
    pea_instance_1d.convergence_threshold = 1.0
    pea_instance_1d.convergence_patience = 1

    best_position, best_fitness = pea_instance_1d.run()

    assert best_position != pytest.approx(3, abs=1e-2)

def test_finds_known_minimum_2d(pea_instance_2d):
    """
    Tests if the algorithm can find the known minimum of the 2D Rosenbrock function.
    """
    best_position, best_fitness = pea_instance_2d.run()

    assert np.allclose(best_position, [1, 1], atol=0.2)
    assert best_fitness == pytest.approx(0, abs=1e-2)

def test_initialization_2d(pea_instance_2d):
    """
    Tests if the 2D algorithm initializes correctly.
    """
    assert pea_instance_2d.obj_function == rosenbrock_function_2d
    assert pea_instance_2d.search_space == [[-2, 2], [-2, 2]]
    assert len(pea_instance_2d.pulses) == 40

def test_pulse_initialization_2d(pea_instance_2d):
    """
    Tests if the 2D pulses are initialized within the search space.
    """
    for pulse in pea_instance_2d.pulses:
        assert len(pulse['center']) == 2
        assert -2 <= pulse['center'][0] <= 2
        assert -2 <= pulse['center'][1] <= 2
        assert pulse['radius'] == 1.0
        assert pulse['best_fitness'] == float('inf')
        assert pulse['best_position'] is None
