"""
    Tests verifying matrix and vector helper utilities using simple reference configurations.
"""

# BAM
from tf_transformations import matrix
from tf_transformations import vector

# PYTHON
import numpy as np
import pytest


def test_xyzrpy_to_matrix_identity():
    result = matrix.xyzrpy_to_matrix()
    assert np.allclose(result, np.eye(4))


def test_matrix_to_xyzrpy_roundtrip():
    translation = (1.0, -2.0, 3.0)
    rpy = (0.1, -0.2, 0.3)
    mat = matrix.xyzrpy_to_matrix(translation, rpy)
    xyz, recovered_rpy = matrix.matrix_to_xyzrpy(mat)
    assert np.allclose(xyz, translation)
    assert np.allclose(recovered_rpy, rpy)


def test_matrix_is_close_within_tolerance():
    base = np.eye(4)
    near = matrix.translate_matrix(base, xyz=(5e-7, 0.0, 0.0))
    assert matrix.matrix_is_close(base, near, pos_tol=1e-6, verbose=False)


def test_matrix_is_close_outside_tolerance():
    base = np.eye(4)
    far = matrix.translate_matrix(base, xyz=(1e-4, 0.0, 0.0))
    assert not matrix.matrix_is_close(base, far, pos_tol=1e-6, verbose=False)


def test_matrix_cartesian_distance():
    start = np.eye(4)
    end = matrix.translate_matrix(start, xyz=(1.0, 2.0, 2.0))
    assert matrix.matrix_cartesian_distance(start, end) == pytest.approx(3.0)


def test_matrix_from_points_orthonormal_basis():
    transform = matrix.matrix_from_points(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert np.allclose(transform, np.eye(4))


def test_xyzrpy_offset_local_vs_global_translation():
    base = matrix.xyzrpy_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2))
    local = matrix.xyzrpy_offset(base, xyz=(1.0, 0.0, 0.0), local=True)
    global_ = matrix.xyzrpy_offset(base, xyz=(1.0, 0.0, 0.0), local=False)
    assert np.allclose(local[:3, 3], [0.0, 1.0, 0.0])
    assert np.allclose(global_[:3, 3], [1.0, 0.0, 0.0])


def test_translate_matrix_global_offset():
    pose = np.eye(4)
    translated = matrix.translate_matrix(pose, xyz=(1.0, -2.0, 3.0))
    assert np.allclose(translated[:3, 3], [1.0, -2.0, 3.0])
    assert np.allclose(translated[:3, :3], np.eye(3))


def test_rotate_matrix_about_z_axis():
    pose = np.eye(4)
    rotated = matrix.rotate_matrix(pose, rpy=(0.0, 0.0, np.pi / 2))
    expected = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(rotated, expected)


def test_vector_normalize_basic():
    vec = np.array([1.0, 1.0, 1.0])
    normalized = vector.normalize(vec)
    assert np.allclose(normalized, vec / np.linalg.norm(vec))


def test_vector_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        vector.normalize(np.zeros(3))


def test_vector_distance():
    assert vector.distance(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_vector_magnitude():
    assert vector.magnitude(np.array([1.0, 2.0, 2.0])) == pytest.approx(3.0)


def test_vector_angle_between_axes():
    angle = vector.angle_between(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert angle == pytest.approx(np.pi / 2)


def test_vector_axis_angle_between_orthogonal():
    axis, angle = vector.axis_angle_between(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0])
    assert angle == pytest.approx(np.pi / 2)


def test_vector_axis_angle_between_opposite():
    axis, angle = vector.axis_angle_between(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0])
    assert angle == pytest.approx(np.pi)
