

from dataclasses import dataclass
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


def _centered_differences(path: np.ndarray, dt: float) -> np.ndarray:
    """Compute centered finite-difference derivative for sampled paths."""
    count, dim = path.shape
    diffs = np.zeros_like(path)

    for idx in range(count):
        if 0 < idx < count - 1:
            diffs[idx] = (path[idx + 1] - path[idx - 1]) / (2.0 * dt)
        elif idx == 0 and count > 1:
            diffs[idx] = (path[idx + 1] - path[idx]) / dt
        elif count > 1:
            diffs[idx] = (path[idx] - path[idx - 1]) / dt

    if count <= 1:
        diffs[:] = 0.0

    return diffs


@dataclass
class CartesianPathMetrics:
    """Cumulative path, velocity, and acceleration metrics for Cartesian paths."""

    t_path: np.ndarray
    path: np.ndarray
    path_d: np.ndarray
    path_dd: np.ndarray
    dimension_names: list[str]


def cartesian_path_analysis(
    matrix_path: np.ndarray,
    t: np.ndarray,
    vec_directions: list[np.ndarray] | np.ndarray | None = None,
) -> CartesianPathMetrics:
    """Compute normalized/raw Cartesian progress metrics plus derivatives."""
    n_points = len(matrix_path)
    if len(t) != n_points:
        raise ValueError(f"Time array length ({len(t)}) must match path length ({n_points})")

    dt = float(np.mean(np.diff(t))) if n_points > 1 else 0.1

    xyz_path = np.zeros(n_points)
    for idx in range(1, n_points):
        xyz_path[idx] = xyz_path[idx - 1] + matrix_cartesian_distance(matrix_path[idx - 1], matrix_path[idx])
    total_xyz = xyz_path[-1]
    xyz_norm = xyz_path / total_xyz if total_xyz > 1e-9 else xyz_path.copy()

    rpy_path = np.zeros(n_points)
    for idx in range(1, n_points):
        q1 = quaternion_from_matrix(matrix_path[idx - 1])
        q2 = quaternion_from_matrix(matrix_path[idx])
        _, theta = quaternion_axangle_diff(q1, q2)
        rpy_path[idx] = rpy_path[idx - 1] + abs(theta)
    total_rpy = rpy_path[-1]
    rpy_norm = rpy_path / total_rpy if total_rpy > 1e-9 else rpy_path.copy()

    xyz_norm_d = _centered_differences(xyz_norm.reshape(-1, 1), dt=dt).flatten()
    rpy_norm_d = _centered_differences(rpy_norm.reshape(-1, 1), dt=dt).flatten()
    xyz_path_d = _centered_differences(xyz_path.reshape(-1, 1), dt=dt).flatten()
    rpy_path_d = _centered_differences(rpy_path.reshape(-1, 1), dt=dt).flatten()

    xyz_norm_dd = _centered_differences(xyz_norm_d.reshape(-1, 1), dt=dt).flatten()
    rpy_norm_dd = _centered_differences(rpy_norm_d.reshape(-1, 1), dt=dt).flatten()
    xyz_path_dd = _centered_differences(xyz_path_d.reshape(-1, 1), dt=dt).flatten()
    rpy_path_dd = _centered_differences(rpy_path_d.reshape(-1, 1), dt=dt).flatten()

    dimension_names = ["XYZ_norm", "RPY_norm", "XYZ", "RPY"]
    path = np.column_stack([xyz_norm, rpy_norm, xyz_path, rpy_path])
    path_d = np.column_stack([xyz_norm_d, rpy_norm_d, xyz_path_d, rpy_path_d])
    path_dd = np.column_stack([xyz_norm_dd, rpy_norm_dd, xyz_path_dd, rpy_path_dd])

    if vec_directions is not None:
        if isinstance(vec_directions, np.ndarray) and vec_directions.ndim == 1:
            vec_iterable = [vec_directions]
        else:
            vec_iterable = list(vec_directions)

        for vec_direction in vec_iterable:
            vec_dir = np.array(vec_direction, dtype=float)
            norm = np.linalg.norm(vec_dir)
            if norm < 1e-9:
                raise ValueError("vec_directions entries must be non-zero vectors")
            vec_dir /= norm

            vec_path = np.zeros(n_points)
            for idx in range(1, n_points):
                displacement = matrix_path[idx][:3, 3] - matrix_path[idx - 1][:3, 3]
                vec_path[idx] = vec_path[idx - 1] + float(np.dot(displacement, vec_dir))

            vec_path_d = _centered_differences(vec_path.reshape(-1, 1), dt=dt).flatten()
            vec_path_dd = _centered_differences(vec_path_d.reshape(-1, 1), dt=dt).flatten()

            path = np.column_stack([path, vec_path])
            path_d = np.column_stack([path_d, vec_path_d])
            path_dd = np.column_stack([path_dd, vec_path_dd])

            vec_str = f"Vec[{vec_dir[0]:.1f},{vec_dir[1]:.1f},{vec_dir[2]:.1f}]"
            dimension_names.append(vec_str)

    return CartesianPathMetrics(
        t_path=t,
        path=path,
        path_d=path_d,
        path_dd=path_dd,
        dimension_names=dimension_names,
    )