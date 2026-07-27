"""
src.transform — HMD-to-World Coordinate Transform Application
=============================================================

Computes and manages coordinate transformations between HMD tracking space and world calibration origin.

This script loads calibration data and computes the transforms necessary to convert between:
  - HMD tracking coordinates (from Unity/HMD tracking system)
  - World coordinates (defined by checkerboard calibration origin)
  - Camera-relative coordinates (from individual ZED cameras)

Key Features:
  - Loads precomputed calibration transforms from JSON files
  - Computes HMD pose relative to world checkerboard origin
  - Handles coordinate system conversions (Unity ↔ OpenCV)
  - Supports multiple calibration modes (barycenter, anchors, floor)
  - Manages rotation matrices and translation vectors
  - Saves computed transforms for downstream use

Usage:
  python -m src.transform
  
Calibration Data Loaded:
  - calibration_results/{mode}/rigid_transform_between_Hf_and_world.json
  - calibration_results/{mode}/world_to_hmd_extrinsics.json
  - Snapshot metadata from snapshot/ directory
  - Camera intrinsics from snapshot JSON

Coordinate Transforms:
  - World↔ Checkerboard: Identity (checkerboard is world origin)
  - Camera↔ World: From calibration_to_world.py
  - HMD↔ World: From hmd_calibration_to_world.py
  - HMD↔ Camera: Composed from above

Main Operations:
  1. Load world-to-HMD transform components (R, t)
  2. Load HMD snapshot metadata and intrinsics
  3. Extract HMD tracking data (position, rotation)
  4. Convert from Unity to OpenCV coordinates
  5. Compute composite transforms
  6. Save results for visualization or downstream processing

Dependencies:
  - utils.utils_calibration (snapshot handling)
  - utils.utils_json (JSON I/O)
  - utils.utils_transform (coordinate conversions)
  - numpy
  - config

Configuration (config.py):
  - CALIBRATION_BOARD_SIZE
  - SQUARE_SIZE

Prerequisites:
  - calibration_to_world.py (world origin established)
  - hmd_calibration_to_world.py (HMD poses computed)
  - Snapshot with metadata in snapshot/ directory

Output:
  - Prints transform matrices to console
  - Can extend to save computed transforms to JSON

Example Output:
  Determinant of M_Hf: 1.00
  Rotation matrices printed to console

Notes:
  - Reflection matrices handle coordinate system handedness
  - Determinant of rotation matrices should be ~1.0 (valid rotation)
  - Transform determinant ~-1.0 indicates reflection (Hf space)
  - All transforms in world-aligned coordinate frame
"""

import os
import cv2
import numpy as np
import sys

# Add the root and src structure to system path to import modules normally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#utils
from utils.utils_transform import unity_to_cv, compute_camera_pose_in_world, quaternion_to_rotation_matrix
from utils.utils_json import load_json, save_json, ensure_dir
from utils.utils_calibration import get_snapshot_files, generate_3d_checkerboard_points
#config
import config 
#
"""
Usage: python -m src.transform
    Compute transforms between Hf space and World space and save them directly 
    to calibration_results/floor/rigid_transform_between_Hf_and_world.json.
"""
  
DISPLAY: bool = False  # Set to True to visualize checkerboard detection and calibration results

# Helper function to extract position and rotation from dict format
def extract_pos_rot(pos_dict, rot_dict):
    pos = [pos_dict["x"], pos_dict["y"], pos_dict["z"]]
    rot = [rot_dict["x"], rot_dict["y"], rot_dict["z"], rot_dict["w"]]
    return pos, rot

if __name__ == "__main__":
    # 1. Get snapshot image and json
    try:
        img_path, json_path = get_snapshot_files("snapshot")
    except Exception as e:
        print(f"[ERROR] {e}")

    print(f"[INFO] Using image: {img_path}")
    print(f"[INFO] Using intrinsics: {json_path}")

    # 2. Parse intrinsics properly 
    meta = load_json(json_path)

    fx = meta["focalLength"]["x"]
    fy = meta["focalLength"]["y"]
    cx = meta["principalPoint"]["x"]
    cy = meta["principalPoint"]["y"]
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assuming no distortion parameters provided in meta, using zero distortion
    dist_coeffs = np.zeros((4, 1))

    # 3. Define 3D checkerboard points
    objp = generate_3d_checkerboard_points(config.CALIBRATION_BOARD_SIZE, config.SQUARE_SIZE)

    # 4. Load Image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] Could not read image {img_path}")
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detect 2D image points
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(frame_gray, config.CALIBRATION_BOARD_SIZE, flags=flags)

    if ret:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
        
        if DISPLAY:
            cv2.drawChessboardCorners(frame, config.CALIBRATION_BOARD_SIZE, corners_refined, ret)
            cv2.imshow("HMD Calibration", frame)
            print("[INFO] Press any key on the image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Compute Extrinsics
        success, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist_coeffs)

        if success:
            R_left_cam, _ = cv2.Rodrigues(rvec)
            R_world_left_cam, t_world_left_cam = compute_camera_pose_in_world(R_left_cam, tvec)

    # Extract Floor anchor data from sceneSpaces
    scene_spaces = meta.get("sceneSpaces", [])
    floor_space = None
    for scene in scene_spaces:
        if scene.get("anchorName") == "Floor":
            floor_space = scene
            break
    
    if floor_space is None:
        print(f"[WARNING] Floor anchor not found in sceneSpaces, skipping floor mode")
    
    data_source = floor_space["poseData"]

    pos_key = "head_position" if "head_position" in data_source else "position"
    rot_key = "head_rotation" if "head_rotation" in data_source else "rotation"

    # --- xrCamera pose estimation ---
    # Transform from left camera to xrCamera in OpenCV space
    pos_xr, rot_xr = extract_pos_rot(data_source["xrCamera_position"], data_source["xrCamera_rotation"])
    pos_left, rot_left = extract_pos_rot(data_source["left_camera_position"], data_source["left_camera_rotation"])

    # flip y axis (left handed to right handed) 
    pos_left_cv, R_left_cv = unity_to_cv(pos_left, rot_left)
    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr, rot_xr) 

    R_left_to_xr = R_left_cv.T @ R_xr_cv
    t_left_to_xr = R_left_cv.T @ (pos_xr_cv - pos_left_cv)

    # Apply to our calibrated left camera poses
    R_world_xr = R_world_left_cam @ R_left_to_xr
    t_world_xr = t_world_left_cam + R_world_left_cam @ t_left_to_xr.reshape(3, 1)

    xr_in_world = {
        "R": R_world_xr.tolist(),
        "t": t_world_xr.tolist()
    }

    R_xr_cam = R_world_xr.T
    t_xr_cam = -R_xr_cam @ t_world_xr

    world_to_xr = {
        "R": R_xr_cam.tolist(),
        "t": t_xr_cam.tolist()
    }

    # 2. Build 4x4 Pose Matrix for HMD in Hf space
    M_raw_left = np.eye(4, dtype=np.float64)
    M_raw_left[:3, :3] = quaternion_to_rotation_matrix(rot_xr)
    M_raw_left[:3, 3] = pos_xr

    # Reflection matrix mirroring the Y-axis inversion applied in unity_to_cv
    S = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0,  1,  0], 
        [0,  0,  0,  1]
    ], dtype=np.float64)

    # Apply the reflection to convert M_Hf into an inverted representation (det = -1.0)
    M_Hf = M_raw_left @ S  
    print(f"Determinant of M_Hf: {np.linalg.det(M_Hf):.2f}") 
    
    M_W = np.eye(4, dtype=np.float64)
    M_W[:3, :3] = R_world_xr
    M_W[:3, 3] = t_world_xr.flatten()

    # 4. Compute Hf-to-World and World-to-Hf transforms
    T_Hf_to_World = M_W @ np.linalg.inv(M_Hf)
    T_World_to_Hf = np.linalg.inv(T_Hf_to_World)

    # 5. Compute the full 4x4 determinant to monitor handedness inversion
    det_Hf_to_World = np.linalg.det(T_Hf_to_World)
    det_World_to_Hf = np.linalg.det(T_World_to_Hf)

    print("\n--- Calibration Space Transforms ---")
    print("\nT_Hf_to_World Matrix:\n", T_Hf_to_World)
    print(f"Determinant of Transform: {det_Hf_to_World:.2f}")

    print("\nT_World_to_Hf Matrix:\n", T_World_to_Hf)
    print(f"Determinant of Transform: {det_World_to_Hf:.2f}")

    # =================================================================
    # Automated JSON Export Setup
    # =================================================================
    output_dir = os.path.join("calibration_results", "floor")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "rigid_transform_between_Hf_and_world.json")

    transform_payload = {
        "Hf_to_world": T_Hf_to_World.tolist(),
        "world_to_Hf": T_World_to_Hf.tolist()
    }

    try:
        save_json(output_path, transform_payload)
        print(f"\n[SUCCESS] Saved calculated spaces accurately to: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save transformation configurations: {e}")