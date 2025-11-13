"""
Ideally you just use the tf_transforms package, but there doesn't seem to be astraight forward pip install...
https://github.com/DLu/tf_transformations

"""
import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm

def distance(vec1: np.ndarray, vec2: np.ndarray):
    return np.linalg.norm(vec2 - vec1)

def magnitude(vec: np.ndarray):
    return np.linalg.norm(vec)

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """ Smallest angle between two vectors
        if parrallel -> 0
        if perpendicular -> 90
        if opposite -> 180
    """

    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cos_theta = dot / (norm_u * norm_v)
    # Clamp to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def axis_angle_between(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the axis and smallest angle to rotate vector u onto vector v.

    DescriptionArgs:
        u, v: Input vectors (any dimension ≥ 3). Only first 3 components matter.

    Returns:
        axis (np.ndarray): Unit vector (3,) perpendicular to both u and v.
        angle (float): Smallest rotation angle in radians.
            - parallel -> 0
            - perpendicular -> pi/2
            - opposite -> pi
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Normalize
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        raise ValueError("Zero-length vector not allowed")
    u /= norm_u
    v /= norm_v

    # Compute angle
    dot = np.dot(u, v)
    cos_theta = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(cos_theta)

    # Compute axis (cross product)
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:  
        # Vectors are parallel or opposite
        if cos_theta > 0:  
            # Same direction → no rotation
            return np.array([1, 0, 0]), 0.0
        else:  
            # Opposite direction → pick arbitrary orthogonal axis
            # Here we pick an axis perpendicular to u
            perp = np.array([1, 0, 0])
            if np.allclose(u, perp):
                perp = np.array([0, 1, 0])
            axis = np.cross(u, perp)
            axis /= np.linalg.norm(axis)
            return axis, np.pi

    axis /= axis_norm
    return axis, angle