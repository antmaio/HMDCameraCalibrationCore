"""
core.transform — Camera Calibration & Coordinate Transformations
==================================================================

Handles loading and computing camera calibration transforms between world and HMD spaces.

This module provides functions for:
  - Loading precomputed world-to-HMD (Hf) rigid transforms from calibration results
  - Building camera projection matrices from intrinsics and extrinsics
  - Loading camera extrinsics from calibration JSON files
  - Accessing calibrated poses for different calibration modes (barycenter, anchors, floor)

Key Features:
  - Loads precomputed calibration transforms from JSON files
  - Handles multiple calibration modes for different tracking spaces
  - Builds 3×4 projection matrices P = K @ [R|t] for camera triangulation
  - Supports accessing both world-to-camera and camera-to-world transforms
  - Validates transform shapes and reports errors clearly

Functions:
  get_world_to_hf_components(mode) -> (R, t): Load world→HMD transform
  get_projection_matrices(cameras) -> proj_matrices: Build projection matrices
  build_projection_matrix(K, R, t) -> P: Construct single projection matrix

Usage:
  # Load world to HMD transforms for specific calibration mode
  R_W_to_Hf, t_W_to_Hf = get_world_to_hf_components('floor')
  
  # Build projection matrices for multi-view triangulation
  proj_mats = get_projection_matrices(cameras)
  
  # Manually construct single projection matrix
  P = build_projection_matrix(K, R, t)

Calibration Modes:
  - 'barycenter': Guardian space center
  - 'anchors': Anchor-based tracking space
  - 'floor': Floor-relative coordinate frame

Dependencies:
  - numpy
  - json
  - utils.utils_calibration (build_intrinsics_from_zed)
  - config (CAMERA_SERIAL_NUMBERS, MIN_KEYPOINT_CONFIDENCE)

Calibration Files:
  - calibration_results/{mode}/rigid_transform_between_Hf_and_world.json
  - calibration_results/world_to_camera_extrinsics.json

Notes:
  - Transforms use (R, t) format where: p_new = R @ p_old + t
  - Projection matrix: 3×4, maps 3D world points to 2D image coordinates
  - Requires camera calibration to be run first (calibration_to_world.py)
"""

import os
import sys
import json
import numpy as np 

#utils
from utils.utils_calibration import build_intrinsics_from_zed
#config
import config 

def get_world_to_hf_components(mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the precomputed World→Hf rigid transform from
    calibration_results/<mode>/rigid_transform_between_Hf_and_world.json
    and return (R_W_to_Hf, t_W_to_Hf) as float32 arrays.
    """
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'calibration_results', mode,
        'rigid_transform_between_Hf_and_world.json'))

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Transform file not found: {path}")
        print("[ERROR] Run 'python -m debug.transform' first to generate it.")
        sys.exit(1)

    T = np.array(data['world_to_Hf'], dtype=np.float32)
    if T.shape != (4, 4):
        print(f"[ERROR] world_to_Hf has unexpected shape {T.shape}.")
        sys.exit(1)

    R = T[:3, :3]
    t = T[:3,  3]
    print(f"[CALIB] Loaded World→Hf transform from {path}")
    return R, t

def get_projection_matrices(cameras: list) -> list:
    """Build projection matrices P = K @ [R | t] for all cameras."""
    extrinsics_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'calibration_results',
        'world_to_camera_extrinsics.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)

    proj_matrices = []
    for cam, sn in zip(cameras, config.CAMERA_SERIAL_NUMBERS):
        cam_info = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = build_intrinsics_from_zed(cam_info).astype(np.float32)

        cam_data = ext.get(f"camera_{sn}")
        if cam_data is None:
            print(f"[ERROR] Extrinsics for camera_{sn} not found in JSON.")
            sys.exit(1)

        R = np.array(cam_data['R'], dtype=np.float32)
        t = np.array(cam_data['t'], dtype=np.float32).reshape(3, 1)

        P = build_projection_matrix(K, R, t)
        proj_matrices.append(P)

    return proj_matrices

def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
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
        proj_matrices.append(build_projection_matrix(K, R, t))
    return proj_matrices
