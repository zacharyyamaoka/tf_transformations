"""
    Simple tests for matrix operations using known values.

    Tests use identity matrices and other predictable inputs where the expected
    results are known.
"""

# PYTHON
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
    # Use a rotation in source to make local vs global different
    source = xyzrpy_to_matrix(xyz=(1.0, 0.0, 0.0), rpy=(np.pi/2, 0.0, 0.0))
    
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 1.0, 0.0]  # Translate in Y direction
    
    result_local = apply_transform_matrix(source, transform, local=True)
    result_global = apply_transform_matrix(source, transform, local=False)
    
    # Results should be different when source has rotation
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

