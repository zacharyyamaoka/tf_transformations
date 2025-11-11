"""
    Simple tests for vector operations using known values.

    Tests use unit vectors and other predictable inputs where the expected
    results are known.
"""

# PYTHON
import numpy as np
import pytest

from tf_transformations.vector import (
    normalize,
    distance,
    magnitude,
    angle_between,
    axis_angle_between,
)


def test_normalize_unit_vectors():
    """Test normalization of unit vectors."""
    # Unit vectors should remain unchanged
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    assert np.allclose(normalize(x_axis), x_axis)
    assert np.allclose(normalize(y_axis), y_axis)
    assert np.allclose(normalize(z_axis), z_axis)


def test_normalize_non_unit_vectors():
    """Test normalization of non-unit vectors."""
    # [1, 1, 1] should normalize to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    vec = np.array([1.0, 1.0, 1.0])
    normalized = normalize(vec)
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert np.allclose(normalized, expected)
    assert np.allclose(np.linalg.norm(normalized), 1.0)


def test_normalize_negative_vector():
    """Test normalization of negative vector."""
    vec = np.array([-1.0, 0.0, 0.0])
    normalized = normalize(vec)
    expected = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(normalized, expected)


def test_normalize_zero_vector():
    """Test that zero vector raises ValueError."""
    vec = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        normalize(vec)


def test_distance_identical_vectors():
    """Test distance between identical vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert np.isclose(distance(vec1, vec2), 0.0)


def test_distance_unit_vectors():
    """Test distance between unit vectors."""
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Distance between perpendicular unit vectors should be sqrt(2)
    assert np.isclose(distance(x_axis, y_axis), np.sqrt(2.0))
    assert np.isclose(distance(x_axis, z_axis), np.sqrt(2.0))
    assert np.isclose(distance(y_axis, z_axis), np.sqrt(2.0))


def test_distance_opposite_vectors():
    """Test distance between opposite vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    assert np.isclose(distance(vec1, vec2), 2.0)


def test_magnitude_unit_vectors():
    """Test magnitude of unit vectors."""
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    assert np.isclose(magnitude(x_axis), 1.0)
    assert np.isclose(magnitude(y_axis), 1.0)
    assert np.isclose(magnitude(z_axis), 1.0)


def test_magnitude_ones_vector():
    """Test magnitude of [1, 1, 1]."""
    vec = np.array([1.0, 1.0, 1.0])
    assert np.isclose(magnitude(vec), np.sqrt(3.0))


def test_angle_between_parallel_vectors():
    """Test angle between parallel vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    angle = angle_between(vec1, vec2)
    assert np.isclose(angle, 0.0)


def test_angle_between_perpendicular_vectors():
    """Test angle between perpendicular vectors."""
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    angle_xy = angle_between(x_axis, y_axis)
    assert np.isclose(angle_xy, np.pi / 2.0)
    
    angle_xz = angle_between(x_axis, z_axis)
    assert np.isclose(angle_xz, np.pi / 2.0)


def test_angle_between_opposite_vectors():
    """Test angle between opposite vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    angle = angle_between(vec1, vec2)
    assert np.isclose(angle, np.pi)


def test_axis_angle_between_parallel_vectors():
    """Test axis-angle between parallel vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    axis, angle = axis_angle_between(vec1, vec2)
    assert np.isclose(angle, 0.0)


def test_axis_angle_between_perpendicular_vectors():
    """Test axis-angle between perpendicular vectors."""
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    
    axis, angle = axis_angle_between(x_axis, y_axis)
    # Should rotate around z-axis
    assert np.allclose(np.abs(axis), np.array([0.0, 0.0, 1.0]))
    assert np.isclose(angle, np.pi / 2.0)


def test_axis_angle_between_opposite_vectors():
    """Test axis-angle between opposite vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    axis, angle = axis_angle_between(vec1, vec2)
    assert np.isclose(angle, np.pi)
    # Axis should be perpendicular to vec1
    assert np.isclose(np.dot(axis, vec1), 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

