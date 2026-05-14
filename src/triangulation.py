"""
triangulation.py
Handles 3D triangulation of 2D keypoints from multiple camera views.

To implement real triangulation, you will need:
  - Intrinsic matrices (K) per camera  →  from ZED calibration parameters
  - Extrinsic matrices (R, t) per camera  →  from an external calibration step
    (e.g., using a ChArUco board with cv2.calibrateCamera / cv2.stereoCalibrate)

Reference:
  - cv2.triangulatePoints for stereo pairs
  - Direct Linear Transformation (DLT) for N≥3 views
"""

import numpy as np
import cv2

NUM_KEYPOINTS = 17


# ---------------------------------------------------------------------------
# Placeholder / Mock
# ---------------------------------------------------------------------------

def triangulate_multiview_mock(
    kpts_cam1: np.ndarray,
    kpts_cam2: np.ndarray,
    kpts_cam3: np.ndarray,
) -> np.ndarray:
    """
    Mock triangulation — returns random 3D coordinates.
    Replace with the real implementation below once calibration data is ready.

    Returns:
        (17, 3) numpy array of 3D keypoints.
    """
    return np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32) * 2.0


# ---------------------------------------------------------------------------
# Real DLT-based triangulation (N views)
# ---------------------------------------------------------------------------

def _build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the 3×4 camera projection matrix P = K @ [R | t].

    Args:
        K: (3, 3) intrinsic matrix.
        R: (3, 3) rotation matrix.
        t: (3, 1) translation vector.

    Returns:
        (3, 4) projection matrix.
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def _dlt_triangulate_point(
    proj_matrices: list[np.ndarray],
    points_2d: list[np.ndarray],
) -> np.ndarray:
    """
    Triangulate a single 3D point from N≥2 2D observations via DLT.

    Args:
        proj_matrices: List of (3, 4) projection matrices.
        points_2d: List of (2,) pixel coordinates, one per camera.

    Returns:
        (3,) array with the triangulated 3D point in world coordinates.
    """
    A = []
    for P, pt in zip(proj_matrices, points_2d):
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)  # shape: (2*N, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # last row = null-space solution
    return (X[:3] / X[3]).astype(np.float32)


def triangulate_multiview(
    kpts_list: list[np.ndarray],
    proj_matrices: list[np.ndarray],
) -> np.ndarray:
    """
    Triangulate all 17 keypoints from N camera views using DLT.

    Args:
        kpts_list: List of (17, 2) arrays, one per camera.
        proj_matrices: List of (3, 4) projection matrices, one per camera.

    Returns:
        (17, 3) numpy array of 3D keypoints in world coordinates.
    """
    pose_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    for kp_idx in range(NUM_KEYPOINTS):
        points_2d = [kpts[kp_idx] for kpts in kpts_list]

        # Skip keypoints that were not detected in any camera
        if all(np.allclose(pt, 0) for pt in points_2d):
            continue

        pose_3d[kp_idx] = _dlt_triangulate_point(proj_matrices, points_2d)

    return pose_3d


def build_projection_matrices(
    intrinsics_list: list[dict],
    extrinsics_list: list[dict],
) -> list[np.ndarray]:
    """
    Convenience function to build all projection matrices.

    Args:
        intrinsics_list: List of dicts with keys 'fx', 'fy', 'cx', 'cy'.
        extrinsics_list: List of dicts with keys 'R' (3×3) and 't' (3,).

    Returns:
        List of (3, 4) projection matrices.
    """
    proj_matrices = []
    for intr, extr in zip(intrinsics_list, extrinsics_list):
        K = np.array([
            [intr["fx"],         0, intr["cx"]],
            [        0, intr["fy"], intr["cy"]],
            [        0,         0,           1],
        ], dtype=np.float64)
        R = np.array(extr["R"], dtype=np.float64)
        t = np.array(extr["t"], dtype=np.float64)
        proj_matrices.append(_build_projection_matrix(K, R, t))
    return proj_matrices
