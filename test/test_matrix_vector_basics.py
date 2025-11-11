"""
    Basic sanity checks for `tf_transformations.matrix` and `tf_transformations.vector`.
"""

# BAM
from tf_transformations.matrix import (
    matrix_cartesian_distance,
    matrix_is_close,
    matrix_to_xyzrpy,
    matrix_from_points,
    xyzrpy_to_matrix,
)
from tf_transformations.vector import (
    angle_between,
    axis_angle_between,
    distance,
    magnitude,
    normalize,
)

# PYTHON
import numpy as np
import pytest


def test_xyzrpy_identity_matrix():
    matrix = xyzrpy_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert np.allclose(matrix, np.eye(4))


def test_matrix_to_xyzrpy_roundtrip():
    xyz = (1.0, -2.0, 3.0)
    rpy = (np.pi / 2, 0.0, np.pi / 4)
    matrix = xyzrpy_to_matrix(xyz, rpy)
    recovered_xyz, recovered_rpy = matrix_to_xyzrpy(matrix)
    assert np.allclose(recovered_xyz, xyz)
    assert np.allclose(recovered_rpy, rpy)


def test_matrix_is_close_and_distance():
    base = xyzrpy_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert matrix_is_close(base, base)

    shifted = xyzrpy_to_matrix((0.01, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert not matrix_is_close(base, shifted, verbose=False)
    assert np.isclose(matrix_cartesian_distance(base, shifted), 0.01)


def test_matrix_from_points_axes():
    matrix = matrix_from_points(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert np.allclose(matrix, np.eye(4))


def test_vector_normalize_and_magnitude():
    vec = np.array([1.0, 1.0, 1.0])
    normalized = normalize(vec)
    assert np.allclose(normalized, vec / np.linalg.norm(vec))
    assert np.isclose(magnitude(vec), np.sqrt(3))


def test_vector_distance_and_angles():
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    opposite_x = np.array([-1.0, 0.0, 0.0])

    assert np.isclose(distance(np.zeros(3), x_axis), 1.0)
    assert np.isclose(angle_between(x_axis, y_axis), np.pi / 2)

    axis, angle = axis_angle_between(x_axis, opposite_x)
    assert np.isclose(angle, np.pi)
    assert np.allclose(np.linalg.norm(axis), 1.0)
    assert np.allclose(axis, np.array([0.0, 0.0, 1.0]))


def test_axis_angle_between_identical_vectors():
    vec = np.array([0.0, 0.0, 1.0])
    axis, angle = axis_angle_between(vec, vec)
    assert np.isclose(angle, 0.0)
    assert np.allclose(axis, np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__])
