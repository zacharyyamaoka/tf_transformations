"""
    Simple tests for matrix and vector operations using known values.

    Tests use identity matrices, unit vectors, and other predictable inputs
    where the expected results are known.
"""
import numpy as np
import pytest

from tf_transformations.matrix import (
    rpy_to_R,
    xyzrpy_to_matrix,
    matrix_to_xyzrpy,
    matrix_is_close,
    matrix_cartesian_distance,
    quaternion_euler_diff,
    quaternion_axangle_diff,
    translate_matrix,
    rotate_matrix,
    apply_transform_matrix,
    xyzrpy_offset,
)
from tf_transformations.vector import (
    normalize,
    distance,
    magnitude,
    angle_between,
    axis_angle_between,
)


# ============================================================================
# VECTOR TESTS
# ============================================================================

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


# ============================================================================
# MATRIX TESTS
# ============================================================================

def test_rpy_to_R_zero_rotation():
    """Test RPY to rotation matrix with zero rotation."""
    rpy = [0.0, 0.0, 0.0]
    R = rpy_to_R(rpy)
    expected = np.eye(3)
    assert np.allclose(R, expected)


def test_rpy_to_R_identity():
    """Test RPY to rotation matrix produces valid rotation matrix."""
    rpy = [0.0, 0.0, 0.0]
    R = rpy_to_R(rpy)
    # Rotation matrix should be orthogonal: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3))
    # Determinant should be 1
    assert np.isclose(np.linalg.det(R), 1.0)


def test_xyzrpy_to_matrix_identity():
    """Test xyzrpy_to_matrix with zero translation and rotation."""
    mat = xyzrpy_to_matrix(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0))
    expected = np.eye(4)
    assert np.allclose(mat, expected)


def test_xyzrpy_to_matrix_translation_only():
    """Test xyzrpy_to_matrix with translation only."""
    xyz = (1.0, 2.0, 3.0)
    rpy = (0.0, 0.0, 0.0)
    mat = xyzrpy_to_matrix(xyz=xyz, rpy=rpy)
    
    # Check translation
    assert np.allclose(mat[:3, 3], xyz)
    # Check rotation is identity
    assert np.allclose(mat[:3, :3], np.eye(3))
    # Check bottom row
    assert np.allclose(mat[3, :], np.array([0.0, 0.0, 0.0, 1.0]))


def test_matrix_to_xyzrpy_identity():
    """Test matrix_to_xyzrpy with identity matrix."""
    mat = np.eye(4)
    xyz, rpy = matrix_to_xyzrpy(mat)
    assert np.allclose(xyz, [0.0, 0.0, 0.0])
    assert np.allclose(rpy, [0.0, 0.0, 0.0])


def test_matrix_to_xyzrpy_roundtrip():
    """Test roundtrip conversion: xyzrpy -> matrix -> xyzrpy."""
    xyz = (1.0, 2.0, 3.0)
    rpy = (0.0, 0.0, 0.0)
    mat = xyzrpy_to_matrix(xyz=xyz, rpy=rpy)
    xyz_out, rpy_out = matrix_to_xyzrpy(mat)
    assert np.allclose(xyz_out, xyz)
    assert np.allclose(rpy_out, rpy)


def test_matrix_is_close_identical():
    """Test matrix_is_close with identical matrices."""
    T1 = np.eye(4)
    T2 = np.eye(4)
    assert matrix_is_close(T1, T2, verbose=False)


def test_matrix_is_close_different_translation():
    """Test matrix_is_close with different translations."""
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [1.0, 0.0, 0.0]
    assert not matrix_is_close(T1, T2, verbose=False)


def test_matrix_cartesian_distance_identical():
    """Test matrix_cartesian_distance with identical matrices."""
    T1 = np.eye(4)
    T2 = np.eye(4)
    dist = matrix_cartesian_distance(T1, T2)
    assert np.isclose(dist, 0.0)


def test_matrix_cartesian_distance_unit_translation():
    """Test matrix_cartesian_distance with unit translation."""
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [1.0, 0.0, 0.0]
    dist = matrix_cartesian_distance(T1, T2)
    assert np.isclose(dist, 1.0)


def test_translate_matrix_identity():
    """Test translate_matrix with identity matrix."""
    pose = np.eye(4)
    result = translate_matrix(pose, xyz=(1.0, 2.0, 3.0), local=False)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(result, expected)


def test_translate_matrix_zero_offset():
    """Test translate_matrix with zero offset."""
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    result = translate_matrix(pose, xyz=(0.0, 0.0, 0.0), local=False)
    assert np.allclose(result, pose)


def test_rotate_matrix_identity():
    """Test rotate_matrix with identity matrix."""
    pose = np.eye(4)
    result = rotate_matrix(pose, rpy=(0.0, 0.0, 0.0), local=False)
    assert np.allclose(result, pose)


def test_apply_transform_matrix_identity():
    """Test apply_transform_matrix with identity matrices."""
    source = np.eye(4)
    transform = np.eye(4)
    result = apply_transform_matrix(source, transform, local=False)
    assert np.allclose(result, np.eye(4))


def test_apply_transform_matrix_local_vs_global():
    """Test apply_transform_matrix local vs global."""
    source = np.eye(4)
    source[:3, 3] = [1.0, 0.0, 0.0]
    
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 1.0, 0.0]
    
    result_local = apply_transform_matrix(source, transform, local=True)
    result_global = apply_transform_matrix(source, transform, local=False)
    
    # Results should be different
    assert not np.allclose(result_local, result_global)
    
    # Global: transform first, then source
    # Local: source first, then transform in local frame
    expected_global = transform @ source
    expected_local = source @ transform
    
    assert np.allclose(result_global, expected_global)
    assert np.allclose(result_local, expected_local)


def test_xyzrpy_offset_zero_offset():
    """Test xyzrpy_offset with zero offset."""
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    result = xyzrpy_offset(matrix, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0))
    assert np.allclose(result, matrix)


def test_xyzrpy_offset_translation():
    """Test xyzrpy_offset with translation only."""
    matrix = np.eye(4)
    result = xyzrpy_offset(matrix, xyz=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, 0.0), local=False)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(result, expected)


def test_quaternion_euler_diff_identical():
    """Test quaternion_euler_diff with identical quaternions."""
    # Identity quaternion: [x, y, z, w] = [0, 0, 0, 1]
    q1 = [0.0, 0.0, 0.0, 1.0]
    q2 = [0.0, 0.0, 0.0, 1.0]
    euler_error = quaternion_euler_diff(q1, q2)
    # Should be close to zero (allowing for numerical errors)
    assert np.allclose(euler_error, [0.0, 0.0, 0.0], atol=1e-6)


def test_quaternion_axangle_diff_identical():
    """Test quaternion_axangle_diff with identical quaternions."""
    # Identity quaternion: [x, y, z, w] = [0, 0, 0, 1]
    q1 = [0.0, 0.0, 0.0, 1.0]
    q2 = [0.0, 0.0, 0.0, 1.0]
    vec, theta = quaternion_axangle_diff(q1, q2)
    # Angle should be zero
    assert np.isclose(theta, 0.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
