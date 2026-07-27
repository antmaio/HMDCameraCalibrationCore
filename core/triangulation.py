"""
core.triangulation — Multi-View 3D Triangulation
===================================================

Handles conversion of 2D multi-camera keypoint observations to 3D world coordinates.

This module implements Direct Linear Transformation (DLT) for triangulating 3D points from
multiple synchronized camera views. It reconstructs the full 3D skeleton from 2D pose
detections across N camera perspectives.

Key Features:
  - Direct Linear Transformation (DLT) algorithm for robust multi-view triangulation
  - Supports N≥2 camera views for redundancy and accuracy
  - Confidence-based filtering: only uses high-confidence keypoint detections
  - Handles missing detections gracefully (0,0 coordinates skipped)
  - Requires pre-computed camera projection matrices (K @ [R|t])
  - Outputs (17, 3) 3D skeleton in world coordinates

Functions:
  triangulate_multiview(kpts_list, proj_matrices) -> pose_3d: Triangulate all keypoints
  _dlt_triangulate_point(proj_matrices, points_2d) -> point_3d: Single point DLT

Usage:
  # kpts_list: List of (17, 3) 2D detections [x, y, conf] from each camera
  # proj_matrices: List of (3, 4) projection matrices P = K @ [R|t]
  pose_3d = triangulate_multiview(kpts_list, proj_matrices)
  # Returns (17, 3) array of 3D keypoints in world coordinates

Algorithm:
  - For each keypoint, collects 2D observations from all cameras where detected
  - Constructs A matrix from projection equations (2 rows per camera)
  - Computes null space via SVD: X = V[-1, :] (last row of V from SVD(A))
  - Converts homogeneous coordinates back to 3D: x = X[:3] / X[3]

Dependencies:
  - numpy
  - config (project configuration, MIN_KEYPOINT_CONFIDENCE, N_COCO_KEYPOINTS)

Prerequisites:
  - Camera calibration files with extrinsics (R, t) relative to world origin
  - Intrinsic matrices (K) from each camera
  - Synchronized 2D pose detections from multiple cameras

Notes:
  - Minimum 2 cameras required; more cameras improve accuracy
  - Invalid detections (0,0 coordinates) are automatically skipped
  - Keypoints below MIN_KEYPOINT_CONFIDENCE threshold are discarded
  - Returns origin (0,0,0) for keypoints not triangulated
"""

import numpy as np
#config
import config


# ---------------------------------------------------------------------------
# Real DLT-based triangulation (N views)
# ---------------------------------------------------------------------------


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
        kpts_list: List of (17, 3) arrays [x, y, conf], one per camera.
        proj_matrices: List of (3, 4) projection matrices, one per camera.

    Returns:
        (17, 3) numpy array of 3D keypoints in world coordinates.
    """
    pose_3d = np.zeros((config.N_COCO_KEYPOINTS, 3), dtype=np.float32)

    for kp_idx in range(config.N_COCO_KEYPOINTS):
        cams_kpt = [kpts[kp_idx] for kpts in kpts_list]  # each (x, y, conf)

        # Only use views where this joint was actually detected
        valid_views = [i for i, pt in enumerate(cams_kpt) if pt[0] != 0 and pt[1] != 0]
        if len(valid_views) < 2:
            continue

        valid_pts   = [cams_kpt[i][:2] for i in valid_views]   # strip conf before DLT
        valid_projs = [proj_matrices[i] for i in valid_views]
        valid_confs = [cams_kpt[i][2]  for i in valid_views]

        # Drop the joint if any contributing view is under-confident
        if min(valid_confs) < config.MIN_KEYPOINT_CONFIDENCE:
            continue

        pose_3d[kp_idx] = _dlt_triangulate_point(valid_projs, valid_pts)

    return pose_3d


