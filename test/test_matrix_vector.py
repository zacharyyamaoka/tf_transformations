"""
    Simple tests for matrix and vector operations using known inputs.

    Tests use identity matrices, unit vectors, and other predictable inputs
    where the expected results are known.
"""

# PYTHON
import numpy as np

# BAM
from tf_transformations import (
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
    distance,
    magnitude,
    angle_between,
    axis_angle_between,
)
from tf_transformations.vector import normalize


# ============================================================================
# VECTOR TESTS
# ============================================================================

def test_normalize_unit_vectors():
    """Test normalize with unit vectors."""
    # Unit vectors should remain unchanged
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    assert np.allclose(normalize(x_axis), x_axis)
    assert np.allclose(normalize(y_axis), y_axis)
    assert np.allclose(normalize(z_axis), z_axis)
    
    # Negative unit vector
    neg_x = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(normalize(neg_x), neg_x)


def test_normalize_non_unit_vectors():
    """Test normalize with non-unit vectors."""
    # [1, 1, 1] should normalize to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    vec = np.array([1.0, 1.0, 1.0])
    normalized = normalize(vec)
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert np.allclose(normalized, expected)
    assert np.allclose(np.linalg.norm(normalized), 1.0)
    
    # [2, 0, 0] should normalize to [1, 0, 0]
    vec2 = np.array([2.0, 0.0, 0.0])
    assert np.allclose(normalize(vec2), [1.0, 0.0, 0.0])


def test_normalize_zero_vector():
    """Test normalize raises error for zero vector."""
    zero_vec = np.array([0.0, 0.0, 0.0])
    try:
        normalize(zero_vec)
        assert False, "Expected ValueError for zero-length vector"
    except ValueError:
        pass  # Expected


def test_distance():
    """Test distance between known vectors."""
    # Distance between [1,0,0] and [0,0,0] should be 1
    assert np.isclose(distance(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])), 1.0)
    
    # Distance between [1,0,0] and [0,1,0] should be sqrt(2)
    assert np.isclose(distance(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])), np.sqrt(2.0))
    
    # Distance between [1,1,1] and [0,0,0] should be sqrt(3)
    assert np.isclose(distance(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])), np.sqrt(3.0))
    
    # Distance between same vectors should be 0
    assert np.isclose(distance(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), 0.0)


def test_magnitude():
    """Test magnitude of known vectors."""
    # Unit vectors should have magnitude 1
    assert np.isclose(magnitude(np.array([1.0, 0.0, 0.0])), 1.0)
    assert np.isclose(magnitude(np.array([0.0, 1.0, 0.0])), 1.0)
    assert np.isclose(magnitude(np.array([0.0, 0.0, 1.0])), 1.0)
    
    # [1,1,1] should have magnitude sqrt(3)
    assert np.isclose(magnitude(np.array([1.0, 1.0, 1.0])), np.sqrt(3.0))
    
    # Zero vector should have magnitude 0
    assert np.isclose(magnitude(np.array([0.0, 0.0, 0.0])), 0.0)
    
    # [-1, 0, 0] should have magnitude 1
    assert np.isclose(magnitude(np.array([-1.0, 0.0, 0.0])), 1.0)


def test_angle_between_parallel():
    """Test angle_between with parallel vectors."""
    # Same direction should give 0 angle
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    assert np.isclose(angle_between(u, v), 0.0)
    
    # Scaled same direction should also give 0
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    assert np.isclose(angle_between(u, v), 0.0)


def test_angle_between_perpendicular():
    """Test angle_between with perpendicular vectors."""
    # [1,0,0] and [0,1,0] should be 90 degrees (pi/2 radians)
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    assert np.isclose(angle_between(u, v), np.pi / 2.0)
    
    # [1,0,0] and [0,0,1] should be 90 degrees
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    assert np.isclose(angle_between(u, v), np.pi / 2.0)


def test_angle_between_opposite():
    """Test angle_between with opposite vectors."""
    # [1,0,0] and [-1,0,0] should be 180 degrees (pi radians)
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([-1.0, 0.0, 0.0])
    assert np.isclose(angle_between(u, v), np.pi)


def test_axis_angle_between_parallel():
    """Test axis_angle_between with parallel vectors."""
    # Same direction should give 0 angle
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    axis, angle = axis_angle_between(u, v)
    assert np.isclose(angle, 0.0)
    
    # Scaled same direction should also give 0
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    axis, angle = axis_angle_between(u, v)
    assert np.isclose(angle, 0.0)


def test_axis_angle_between_perpendicular():
    """Test axis_angle_between with perpendicular vectors."""
    # [1,0,0] and [0,1,0] should give z-axis and pi/2 angle
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    axis, angle = axis_angle_between(u, v)
    assert np.isclose(angle, np.pi / 2.0)
    # Axis should be z-axis (or negative z-axis)
    assert np.allclose(np.abs(axis), [0.0, 0.0, 1.0])


def test_axis_angle_between_opposite():
    """Test axis_angle_between with opposite vectors."""
    # [1,0,0] and [-1,0,0] should give pi angle
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([-1.0, 0.0, 0.0])
    axis, angle = axis_angle_between(u, v)
    assert np.isclose(angle, np.pi)
    # Axis should be perpendicular to u
    assert np.isclose(np.dot(axis, u), 0.0)


def test_axis_angle_between_zero_vector():
    """Test axis_angle_between raises error for zero vector."""
    u = np.array([1.0, 0.0, 0.0])
    zero_vec = np.array([0.0, 0.0, 0.0])
    try:
        axis_angle_between(u, zero_vec)
        assert False, "Expected ValueError for zero-length vector"
    except ValueError:
        pass  # Expected
    try:
        axis_angle_between(zero_vec, u)
        assert False, "Expected ValueError for zero-length vector"
    except ValueError:
        pass  # Expected


# ============================================================================
# MATRIX TESTS
# ============================================================================

def test_rpy_to_R_zero_rotation():
    """Test rpy_to_R with zero rotation (should be identity)."""
    rpy = [0.0, 0.0, 0.0]
    R = rpy_to_R(rpy)
    expected = np.eye(3)
    assert np.allclose(R, expected)


def test_xyzrpy_to_matrix_identity():
    """Test xyzrpy_to_matrix with zero translation and rotation."""
    mat = xyzrpy_to_matrix(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0))
    expected = np.eye(4)
    assert np.allclose(mat, expected)


def test_xyzrpy_to_matrix_translation():
    """Test xyzrpy_to_matrix with translation only."""
    mat = xyzrpy_to_matrix(xyz=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, 0.0))
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(mat, expected)


def test_matrix_to_xyzrpy_identity():
    """Test matrix_to_xyzrpy with identity matrix."""
    identity = np.eye(4)
    xyz, rpy = matrix_to_xyzrpy(identity)
    assert np.allclose(xyz, [0.0, 0.0, 0.0])
    assert np.allclose(rpy, [0.0, 0.0, 0.0])


def test_matrix_to_xyzrpy_round_trip():
    """Test round-trip conversion: xyzrpy -> matrix -> xyzrpy."""
    xyz_in = [1.0, 2.0, 3.0]
    rpy_in = [0.0, 0.0, 0.0]
    mat = xyzrpy_to_matrix(xyz_in, rpy_in)
    xyz_out, rpy_out = matrix_to_xyzrpy(mat)
    assert np.allclose(xyz_out, xyz_in)
    assert np.allclose(rpy_out, rpy_in)


def test_matrix_is_close_identical():
    """Test matrix_is_close with identical matrices."""
    identity = np.eye(4)
    assert matrix_is_close(identity, identity, verbose=False)


def test_matrix_is_close_different():
    """Test matrix_is_close with different matrices."""
    identity = np.eye(4)
    translated = np.eye(4)
    translated[:3, 3] = [10.0, 10.0, 10.0]
    assert not matrix_is_close(identity, translated, verbose=False)


def test_matrix_cartesian_distance():
    """Test matrix_cartesian_distance with known positions."""
    T1 = np.eye(4)
    T1[:3, 3] = [0.0, 0.0, 0.0]
    
    T2 = np.eye(4)
    T2[:3, 3] = [1.0, 0.0, 0.0]
    
    # Distance should be 1
    assert np.isclose(matrix_cartesian_distance(T1, T2), 1.0)
    
    # Distance between [0,0,0] and [1,1,1] should be sqrt(3)
    T3 = np.eye(4)
    T3[:3, 3] = [1.0, 1.0, 1.0]
    assert np.isclose(matrix_cartesian_distance(T1, T3), np.sqrt(3.0))


def test_quaternion_euler_diff_same():
    """Test quaternion_euler_diff with same quaternion (should be zero)."""
    # Identity quaternion [0, 0, 0, 1]
    q = [0.0, 0.0, 0.0, 1.0]
    euler_diff = quaternion_euler_diff(q, q)
    assert np.allclose(euler_diff, [0.0, 0.0, 0.0])


def test_quaternion_axangle_diff_same():
    """Test quaternion_axangle_diff with same quaternion (should be zero angle)."""
    # Identity quaternion [0, 0, 0, 1]
    q = [0.0, 0.0, 0.0, 1.0]
    vec, theta = quaternion_axangle_diff(q, q)
    assert np.isclose(theta, 0.0)


def test_translate_matrix_identity():
    """Test translate_matrix with identity matrix."""
    identity = np.eye(4)
    # Translate by [1, 2, 3]
    result = translate_matrix(identity, xyz=(1.0, 2.0, 3.0), local=False)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(result, expected)


def test_translate_matrix_zero():
    """Test translate_matrix with zero translation (should be unchanged)."""
    identity = np.eye(4)
    result = translate_matrix(identity, xyz=(0.0, 0.0, 0.0), local=False)
    assert np.allclose(result, identity)


def test_rotate_matrix_identity():
    """Test rotate_matrix with identity matrix and zero rotation."""
    identity = np.eye(4)
    result = rotate_matrix(identity, rpy=(0.0, 0.0, 0.0), local=False)
    assert np.allclose(result, identity)


def test_apply_transform_matrix_identity():
    """Test apply_transform_matrix with identity matrices."""
    identity = np.eye(4)
    # Applying identity transform should give identity
    result = apply_transform_matrix(identity, identity, local=False)
    assert np.allclose(result, identity)
    
    result = apply_transform_matrix(identity, identity, local=True)
    assert np.allclose(result, identity)


def test_apply_transform_matrix_translation_global():
    """Test apply_transform_matrix with translation in global frame."""
    identity = np.eye(4)
    translation = np.eye(4)
    translation[:3, 3] = [1.0, 0.0, 0.0]
    
    result = apply_transform_matrix(identity, translation, local=False)
    assert np.allclose(result[:3, 3], [1.0, 0.0, 0.0])


def test_xyzrpy_offset_zero():
    """Test xyzrpy_offset with zero offset (should be unchanged)."""
    identity = np.eye(4)
    result = xyzrpy_offset(identity, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), local=False)
    assert np.allclose(result, identity)


def test_xyzrpy_offset_translation():
    """Test xyzrpy_offset with translation only."""
    identity = np.eye(4)
    result = xyzrpy_offset(identity, xyz=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, 0.0), local=False)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(result, expected)


if __name__ == "__main__":
    # Run all test functions
    import sys
    test_functions = [name for name in dir() if name.startswith("test_")]
    failed = 0
    for test_name in test_functions:
        try:
            globals()[test_name]()
            print(f"✓ {test_name}")
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: {type(e).__name__}: {e}")
            failed += 1
    
    if failed == 0:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
    else:
        print(f"\n{failed} out of {len(test_functions)} tests failed.")
        sys.exit(1)
