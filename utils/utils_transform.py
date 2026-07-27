"""
utils.utils_transform — Coordinate Space Transformations
===========================================================

Provides utilities for coordinate space conversions between Unity and OpenCV.

This module contains functions for:
  - Converting quaternions to rotation matrices
  - Transforming coordinates between Unity (left-handed) and OpenCV (right-handed) spaces
  - Computing camera poses in world frame from world-to-camera extrinsics
  - Handling coordinate system differences (Y-axis flip)

Key Features:
  - Quaternion to rotation matrix conversion with normalization
  - Bi-directional Unity ↔ OpenCV coordinate transforms
  - Handles Y-axis reflection required by coordinate system handedness
  - Computes camera-in-world pose from standard extrinsics
  - Vectorized operations for batch transformations

Functions:
  quaternion_to_rotation_matrix(q) -> R: Convert quaternion [x,y,z,w] to 3×3 rotation matrix
  unity_to_cv(pos, rot_q) -> (pos_cv, R_cv): Transform pose from Unity to OpenCV space
  cv_to_unity(pos_cv) -> pos_unity: Transform position from OpenCV to Unity space
  compute_camera_pose_in_world(R, t) -> (R_world, t_world): Invert extrinsics to get camera-in-world

Usage:
  # Quaternion to rotation matrix
  q = np.array([0, 0, 0.707, 0.707])  # 90° rotation around Z
  R = quaternion_to_rotation_matrix(q)  # 3×3 matrix
  
  # Convert Unity pose to OpenCV
  pos_unity = np.array([1.0, 2.0, 3.0])
  rot_unity_q = np.array([0, 0, 0, 1])  # identity quaternion
  pos_cv, R_cv = unity_to_cv(pos_unity, rot_unity_q)
  
  # Compute camera position in world
  R_world, t_world = compute_camera_pose_in_world(R_camera, t_camera)
  
  # Convert back to Unity
  pos_unity_again = cv_to_unity(pos_cv)

Coordinate Systems:
  Unity (Left-handed):  X right, Y up, Z back
  OpenCV (Right-handed): X right, Y down, Z forward
  
  Transformation: Y-axis flip (Y -> -Y)
  Reflection matrix M = diag(1, -1, 1)

Quaternion Format:
  [x, y, z, w] where w is scalar (XYZW convention)
  
Dependencies:
  - numpy

Notes:
  - Quaternions are normalized before conversion
  - Y-axis flip is bidirectional (Z = -Z in reverse)
  - Works with single vectors and batch arrays (batch operations in cv_to_unity)
  - Camera extrinsics follow OpenCV convention: world-to-camera transformation
"""
import numpy as np

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def unity_to_cv(pos: np.ndarray, rot_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Unity world pose to OpenCV coordinate space pose."""
    pos = np.asarray(pos, dtype=np.float64)
    rot_q = np.asarray(rot_q, dtype=np.float64)
    pos_cv = np.array([pos[0], -pos[1], pos[2]], dtype=np.float64)
    R_u = quaternion_to_rotation_matrix(rot_q)
    M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    R_cv = M @ R_u @ M
    return pos_cv, R_cv


def cv_to_unity(pos_cv: np.ndarray) -> np.ndarray:
    """Convert a position from OpenCV coordinate space back to Unity coordinate space."""
    pos_cv = np.asarray(pos_cv, dtype=np.float64)
    if pos_cv.ndim == 1:
        return np.array([pos_cv[0], -pos_cv[1], pos_cv[2]], dtype=np.float32)
    out = pos_cv.copy().astype(np.float32)
    out[:, 1] = -out[:, 1]
    return out


def compute_camera_pose_in_world(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the camera pose in the world frame from world-to-camera extrinsics."""
    R_world = R.T
    t_world = -R_world @ t.reshape(3, 1)
    return R_world, t_world
