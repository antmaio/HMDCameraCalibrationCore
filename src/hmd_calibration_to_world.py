"""
src.hmd_calibration_to_world — HMD Pose Calibration
====================================================

Estimates HMD and XR camera poses in world coordinates from a snapshot image.

This script takes a pre-recorded snapshot (image + JSON metadata from HMD) containing:
  - A visible checkerboard pattern (for absolute pose estimation)
  - HMD tracking data (position and orientation)
  - Camera intrinsics from the HMD's left camera

It computes where the HMD's head, cameras, and tracking references are in the
world coordinate frame established by calibration_to_world.py.

Key Features:
  - Loads HMD intrinsics and tracking data from JSON snapshot
  - Detects checkerboard in snapshot to establish world pose
  - Computes HMD head pose relative to world checkerboard origin
  - Estimates left camera and xrCamera poses in world coordinates
  - Supports multiple calibration modes (barycenter, anchors, floor)
  - Converts HMD tracking data from Unity to OpenCV coordinates
  - Saves both world-to-HMD and HMD-to-world transforms

Command-Line Arguments:
  --mode {barycenter,anchors,floor,both}  Calibration mode (required)
  --display                               Show checkerboard detection results
  --snapshot_dir                          Directory containing snapshot files
  --out_dir                               Output directory for calibration results

Usage:
  # Calibrate HMD in floor mode
  python -m src.hmd_calibration_to_world --mode floor
  
  # All modes at once
  python -m src.hmd_calibration_to_world --mode both --display
  
  # Custom directories
  python -m src.hmd_calibration_to_world --mode floor \
    --snapshot_dir my_snapshots --out_dir my_results

Calibration Workflow:
  1. Load latest snapshot image and JSON metadata
  2. Extract HMD intrinsics from JSON
  3. Generate 3D checkerboard points (world origin)
  4. Detect checkerboard corners in snapshot image
  5. Refine corners to sub-pixel accuracy
  6. Solve PnP to get left camera pose relative to checkerboard
  7. Load HMD tracking data from JSON for requested mode
  8. Convert Unity tracking poses to OpenCV coordinates
  9. Compute relative transforms between cameras
  10. Apply calibrated left camera pose to get head and xrCamera poses in world
  11. Save transforms for each mode

Calibration Modes:
  barycenter: Guardian space center (center of playable area)
  anchors:    Anchor-based tracking (spatial anchors reference frame)
  floor:      Floor anchor (floor plane reference)
  both:       Save calibration for all three modes

Output Files (per mode):
  calibration_results/{mode}/world_to_hmd_extrinsics.json
    - hmd: World-to-head transform
    - left_eye: World-to-left camera
    - xr_camera: World-to-xrCamera
    
  calibration_results/{mode}/hmd_poses_in_world.json
    - hmd: Head position and orientation in world
    - left_eye: Left camera pose in world
    - xr_camera: XR camera pose in world

JSON Format (Example):
  world_to_hmd_extrinsics.json:
  {
    "hmd": {"R": [[...]], "t": [...]},
    "left_eye": {"R": [[...]], "t": [...]},
    "xr_camera": {"R": [[...]], "t": [...]}  
  }

Snapshot File Format:
  Snapshot_TIMESTAMP.png - HMD camera image with checkerboard visible
  Snapshot_TIMESTAMP.json - Metadata containing:
    - focalLength: {x, y} - Camera focal length in pixels
    - principalPoint: {x, y} - Camera principal point in pixels
    - sceneSpaces: [list of tracking spaces]
      - For each space: poseData with positions/rotations

Coordinate Systems:
  Unity (HMD tracking): Left-handed, Y-up, Z-back
  OpenCV (calibration): Right-handed, Y-down, Z-forward
  World (checkerboard): Defined by checkerboard at origin

Dependencies:
  - utils.utils_calibration (checkerboard generation, snapshot finding)
  - utils.utils_json (JSON I/O)
  - utils.utils_transform (coordinate conversions)
  - cv2 (checkerboard detection, PnP solving)
  - numpy
  - config

Prerequisites:
  - calibration_to_world.py must be run first (establishes world origin)
  - HMD must capture snapshot with checkerboard visible
  - Board size and square size must match calibration_to_world.py

Related Functions:
  - extract_pos_rot(): Parse position/rotation from JSON dict
  - Handles different JSON key names (head_position vs. position, etc.)

Next Steps:
  - Use world-to-HMD transforms in send_yolo_positions.py
  - Select calibration mode matching your HMD's tracking space
"""

import cv2
import numpy as np
import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#utlis
from utils.utils_calibration import generate_3d_checkerboard_points, get_snapshot_files
from utils.utils_json import load_json, save_json, ensure_dir
from utils.utils_transform import unity_to_cv, compute_camera_pose_in_world
#config
import config

def main():
    """Run HMD calibration from a snapshot image and save world pose data for the requested mode."""
    parser = argparse.ArgumentParser(description="HMD Camera Calibration to Common World Origin")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors", "floor", "both"], required=True, help="Calibration mode: barycenter, anchors, floor, or both")
    parser.add_argument("--display", action='store_true', help="Display calibration images")
    parser.add_argument("--snapshot_dir", type=str, default="snapshot", help="Directory with Snapshot files")
    parser.add_argument("--out_dir", type=str, default="calibration_results", help="Base directory to save calibration results")
    args = parser.parse_args()

    # 1. Get snapshot image and json
    try:
        img_path, json_path = get_snapshot_files(args.snapshot_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

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
        return
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detect 2D image points
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(frame_gray, config.CALIBRATION_BOARD_SIZE, flags=flags)

    if ret:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
        
        if args.display:
            cv2.drawChessboardCorners(frame, config.CALIBRATION_BOARD_SIZE, corners_refined, ret)
            cv2.imshow("HMD Calibration", frame)
            print("[INFO] Press any key on the image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Compute Extrinsics
        success, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist_coeffs)

        if success:
            R_left_cam, _ = cv2.Rodrigues(rvec)
            
            world_to_left_cam = {
                "R": R_left_cam.tolist(),
                "t": tvec.tolist()
            }

            R_world_left_cam, t_world_left_cam = compute_camera_pose_in_world(R_left_cam, tvec)
            left_cam_in_world = {
                "R": R_world_left_cam.tolist(),
                "t": t_world_left_cam.tolist()
            }
            
            if args.mode == "both":
                modes_to_test = [("barycenter", "guardianSpace"), ("anchors", "anchorSpace"), ("floor", "sceneSpaces")]
            else:
                modes_to_test = [(args.mode, None)]

            for target_mode, space_key in modes_to_test:
                # Handle different data sources
                if target_mode == "floor":
                    # Extract Floor anchor data from sceneSpaces
                    scene_spaces = meta.get("sceneSpaces", [])
                    floor_space = None
                    for scene in scene_spaces:
                        if scene.get("anchorName") == "Floor":
                            floor_space = scene
                            break
                    
                    if floor_space is None:
                        print(f"[WARNING] Floor anchor not found in sceneSpaces, skipping floor mode")
                        continue
                    
                    data_source = floor_space["poseData"]
                else:
                    data_source = meta[space_key] if space_key else meta

                pos_key = "head_position" if "head_position" in data_source else "position"
                rot_key = "head_rotation" if "head_rotation" in data_source else "rotation"

                # Helper function to extract position and rotation from dict format
                def extract_pos_rot(pos_dict, rot_dict):
                    pos = [pos_dict["x"], pos_dict["y"], pos_dict["z"]]
                    rot = [rot_dict["x"], rot_dict["y"], rot_dict["z"], rot_dict["w"]]
                    return pos, rot

                # --- Head pose estimation ---
                # Unity coordinates converted to OpenCV space before relative transformation math
                pos_head, rot_head = extract_pos_rot(data_source[pos_key], data_source[rot_key])
                pos_head_cv, R_head_cv = unity_to_cv(pos_head, rot_head)
                pos_left, rot_left = extract_pos_rot(data_source["left_camera_position"], data_source["left_camera_rotation"])
                pos_left_cv, R_left_cv = unity_to_cv(pos_left, rot_left)
                
                # Transform from left camera to head center in OpenCV Space
                R_left_to_head = R_left_cv.T @ R_head_cv
                t_left_to_head = R_left_cv.T @ (pos_head_cv - pos_left_cv)
                
                # Apply to our calibrated left camera poses
                # OpenCV Head pose in world = OpenCV Left Cam pose in world * CV Left Cam to Head Offset
                R_world_head = R_world_left_cam @ R_left_to_head
                t_world_head = t_world_left_cam + R_world_left_cam @ t_left_to_head.reshape(3, 1)
                
                head_in_world = {
                    "R": R_world_head.tolist(),
                    "t": t_world_head.tolist()
                }
                
                R_head_cam = R_world_head.T
                t_head_cam = -R_head_cam @ t_world_head
                
                world_to_head = {
                    "R": R_head_cam.tolist(),
                    "t": t_head_cam.tolist()
                }

                # --- xrCamera pose estimation ---
                # Transform from left camera to xrCamera in OpenCV space
                pos_xr, rot_xr = extract_pos_rot(data_source["xrCamera_position"], data_source["xrCamera_rotation"])
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

                target_out_dir = os.path.join(args.out_dir, target_mode)
                ensure_dir(target_out_dir)
                
                save_path_ext = os.path.join(target_out_dir, "world_to_hmd_extrinsics.json")
                save_path_pose = os.path.join(target_out_dir, "hmd_poses_in_world.json")
                
                save_json(save_path_ext, {
                    "hmd": world_to_head, 
                    "left_eye": world_to_left_cam,
                    "xr_camera": world_to_xr
                })
                
                save_json(save_path_pose, {
                    "hmd": head_in_world, 
                    "left_eye": left_cam_in_world,
                    "xr_camera": xr_in_world
                })  

                print(f'\n[INFO] ---- HMD Calibration Results (Common World Origin) - Mode: {target_mode.upper()} ----')
                print("Physical Head Pose in World Frame:")
                print("R:\n", np.array(head_in_world["R"]))
                print("t:\n", np.array(head_in_world["t"]))
                print("\nPhysical Left Eye Pose in World Frame:")
                print("R:\n", np.array(left_cam_in_world["R"]))
                print("t:\n", np.array(left_cam_in_world["t"]))
                print("\nPhysical xrCamera Pose in World Frame:")
                print("R:\n", np.array(xr_in_world["R"]))
                print("t:\n", np.array(xr_in_world["t"]))
                print('[INFO] Calibration files saved to:', target_out_dir)
                
            print("\n[INFO] HMD extrinsics successfully computed and saved.")
            
    else:
        print("\033[91m[ERROR] Checkerboard not detected in the HMD image. Please ensure the snapshot contains a visible checkerboard.\033[0m")
        if args.display:
            cv2.imshow("HMD Calibration [FAILED]", frame)
            print("[INFO] Press any key on the image window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
