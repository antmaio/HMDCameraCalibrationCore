import cv2
import numpy as np
import os
import argparse
import json
import glob
import sys
from scipy.spatial.transform import Rotation as R

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def generate_3d_checkerboard_points(board_size, square_size):
    """
    Generate 3D coordinates of the checkerboard inner corners (Z=0).
    This establishes the World Coordinate System origin at the first inner corner.
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def get_snapshot_files(snapshot_dir):
    png_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.png"))
    if not png_files:
        raise FileNotFoundError(f"No Snapshot_*.png found in {snapshot_dir}")
    # Use the most recent snapshot based on file name or modification time
    img_path = sorted(png_files)[-1]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(snapshot_dir, f"{base_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON file {json_path} for {img_path}")
    return img_path, json_path

def unity_to_cv(pos_dict, rot_dict):
    pos = np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]])
    rot_q = [rot_dict["x"], rot_dict["y"], rot_dict["z"], rot_dict["w"]]
    pos_cv = np.array([pos[0], -pos[1], pos[2]])
    R_u = R.from_quat(rot_q).as_matrix()
    M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_cv = M @ R_u @ M
    return pos_cv, R_cv

def main():
    parser = argparse.ArgumentParser(description="HMD Camera Calibration to Common World Origin")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors", "both"], required=True, help="Calibration mode: barycenter, anchors, or both")
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
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
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

            R_world_left_cam = R_left_cam.T
            t_world_left_cam = -R_world_left_cam @ tvec
            
            left_cam_in_world = {
                "R": R_world_left_cam.tolist(),
                "t": t_world_left_cam.tolist()
            }
            
            if args.mode == "both":
                modes_to_test = [("barycenter", "guardianSpace"), ("anchors", "anchorSpace")]
            else:
                modes_to_test = [(args.mode, None)]

            for target_mode, space_key in modes_to_test:
                data_source = meta[space_key] if space_key else meta

                pos_key = "head_position" if "head_position" in data_source else "position"
                rot_key = "head_rotation" if "head_rotation" in data_source else "rotation"

                # --- Head pose estimation ---
                # Unity coordinates converted to OpenCV space before relative transformation math
                pos_head_cv, R_head_cv = unity_to_cv(data_source[pos_key], data_source[rot_key])
                pos_left_cv, R_left_cv = unity_to_cv(data_source["left_camera_position"], data_source["left_camera_rotation"])
                
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
                pos_xr_cv, R_xr_cv = unity_to_cv(data_source["xrCamera_position"], data_source["xrCamera_rotation"])

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
                os.makedirs(target_out_dir, exist_ok=True)
                
                save_path_ext = os.path.join(target_out_dir, "world_to_hmd_extrinsics.json")
                save_path_pose = os.path.join(target_out_dir, "hmd_poses_in_world.json")
                
                with open(save_path_ext, "w") as f:
                    json.dump({
                        "hmd": world_to_head, 
                        "left_eye": world_to_left_cam,
                        "xr_camera": world_to_xr
                    }, f, indent=4)
                    
                with open(save_path_pose, "w") as f:
                    json.dump({
                        "hmd": head_in_world, 
                        "left_eye": left_cam_in_world,
                        "xr_camera": xr_in_world
                    }, f, indent=4)

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
