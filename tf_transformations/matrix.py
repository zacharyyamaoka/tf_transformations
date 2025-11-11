

from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import quat2axangle 
from tf_transformations import (
    euler_from_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    _reorder_input_quaternion,
    euler_matrix,
    quaternion_from_matrix,
)
import numpy as np

import copy

# CONVERSIONS

# Not implemented in tf_transforms, so use from transforms3d. Careful for quat ordering though
def quaternion_to_axangle(quaterion: list | tuple | np.ndarray):
    # use identity threshold of 1e-8, for when checking pose differences on IK, etc..
    vec, theta = quat2axangle(_reorder_input_quaternion(quaterion), 1e-8) #big bug! (x, y, z, w) ROS -> (w, x, y, z) transforms3d 
    return vec, theta

def rpy_to_R(rpy: list | tuple | np.ndarray) -> np.ndarray:
    return euler2mat(*rpy)

def xyzrpy_to_matrix(
    xyz: list | tuple = (0.0, 0.0, 0.0),
    rpy: list | tuple = (0.0, 0.0, 0.0),
):
    mat = np.eye(4)
    mat[:3, :3] = euler2mat(*rpy)  # rotation
    mat[:3, 3] = xyz               # translation
    return mat

def matrix_to_xyzrpy(mat: np.ndarray) -> tuple[list[float], list[float]]:
    xyz = mat[:3, 3].tolist()
    rpy = list(mat2euler(mat[:3, :3]))
    return xyz, rpy


# Calculate angle between two quaternions...
# https://www.mathworks.com/matlabcentral/answers/476474-how-to-find-the-angle-between-two-quaternions
# https://stackoverflow.com/questions/23260939/distance-or-angular-magnitude-between-two-quaternions
# See also moveit servo


# DISTANCES

def matrix_is_close(T1: np.ndarray,
                    T2: np.ndarray,
                    pos_tol = 1e-6,
                    theta_tol = 1e-3,
                    verbose = True) -> bool:

    xyz1 = T1[:3, 3]
    xyz2 = T2[:3, 3]
    q1 = quaternion_from_matrix(T1)
    q2 = quaternion_from_matrix(T2)

    pos_diff = np.linalg.norm(xyz2 - xyz1)

    vec, theta_diff = quaternion_axangle_diff(q1, q2)

    # Check direct comparison
    xyz_ok = abs(pos_diff) < pos_tol
    rpy_ok = abs(theta_diff) < theta_tol

    # If not OK, check for quaternion sign ambiguity and angle wrapping
    if not rpy_ok:
        # Quaternion sign ambiguity: q and -q represent the same rotation
        # If theta_diff is close to ±π, check if it's actually the same orientation
        if abs(abs(theta_diff) - np.pi) < theta_tol:
            rpy_ok = True
        # Also try ±2π wrapping check
        elif any(abs(theta_diff + shift) < theta_tol for shift in (-2 * np.pi, 2 * np.pi)):
            rpy_ok = True

    if xyz_ok and rpy_ok:
        return True
    else:
        if verbose:
            print(f"XYZ error: {pos_diff} > {pos_tol}")
            print(f"RPY error: {theta_diff} > {theta_tol}")
        return False

def matrix_cartesian_distance(T1: np.ndarray, T2: np.ndarray):
    return np.linalg.norm(T2[:3, 3] - T1[:3, 3])

def quaternion_euler_diff(q1: list | tuple | np.ndarray, q2: list | tuple | np.ndarray):
    q_diff = quaternion_multiply(q2, quaternion_inverse(q1))
    euler_error = np.array(euler_from_quaternion(q_diff))
    return euler_error

def quaternion_axangle_diff(q1: list | tuple | np.ndarray, q2: list | tuple | np.ndarray):
    q_diff = quaternion_multiply(q2, quaternion_inverse(q1))
    vec, theta = quaternion_to_axangle(q_diff)
    return vec, theta


def matrix_from_points(p1: list | tuple | np.ndarray, p2: list | tuple | np.ndarray, p3: list | tuple | np.ndarray):

    """
    p1 = Origin
    p2 - p1 = x-axis
    p3 - p1 = y-axis
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    T = np.eye(4)

    T[:3,3] = p1

    x_axis = p2 - p1
    print("X axis:")
    print(x_axis)
    x_axis /= np.linalg.norm(x_axis)

    temp_y_axis = p3 - p1
    print("Temp Y Axis:")
    print(temp_y_axis)
    z_axis = np.cross(x_axis, temp_y_axis)  
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)  
    y_axis /= np.linalg.norm(y_axis)

    print("Orthogonal Check:")
    print(z_axis @ x_axis)
    print(z_axis @ y_axis)
    print(x_axis @ y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    print("Rotation Matrix:")
    print(R)

    T[:3, :3] = R 

    return T

# TRANSFORMATIONS

def xyzrpy_offset(matrix, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), local=False) -> np.ndarray:
    source_matrix = copy.deepcopy(matrix)
    offset_matrix = xyzrpy_to_matrix(xyz, rpy)
    result_matrix = apply_transform_matrix(source_matrix, offset_matrix, local)
    return result_matrix
    

def translate_matrix(pose_matrix: np.ndarray, xyz=(0.0, 0.0, 0.0), local=False) -> np.ndarray:
    """
    Translate a pose matrix by a given xyz offset.
    """
    T = np.eye(4)
    T[:3, 3] = np.array(xyz)
    return apply_transform_matrix(pose_matrix, T, local)

def rotate_matrix(pose_matrix: np.ndarray, rpy=(0.0, 0.0, 0.0), local=False) -> np.ndarray:
    """
    Rotate a pose matrix by a given rpy offset.
    """
    T = euler_matrix(rpy[0], rpy[1], rpy[2])
    return apply_transform_matrix(pose_matrix, T, local)

def apply_transform_matrix(source_matrix: np.ndarray, transform_matrix: np.ndarray, local=False) -> np.ndarray:
    """
    Local = True means: "First apply the pose, then apply the offset in that local frame"
    Local = False means: "Apply the offset first in global/world frame, then apply the pose transform."
    """

    if local:
        result_matrix = np.dot(source_matrix, transform_matrix)  # Local offset
    else:
        result_matrix = np.dot(transform_matrix, source_matrix)  # Global offset

    return result_matrix