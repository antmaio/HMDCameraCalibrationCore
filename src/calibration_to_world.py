# External
import pyzed.sl as sl
import cv2 
import numpy as np
import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

# Internal
import config
from src.camera import init_cameras, grab_frames, close_cameras

def generate_3d_checkerboard_points(board_size, square_size):
    """
    Generate 3D coordinates of the checkerboard inner corners (Z=0).
    This establishes the World Coordinate System origin at the first inner corner.
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def main():
    parser = argparse.ArgumentParser(description="ZED Camera Calibration to Common World Origin")
    parser.add_argument("--display", action='store_true', help="Display intermediate calibration images, useful for debug")
    parser.add_argument("--out_dir", type=str, default="calibration_results", help="Directory to save calibration results")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Initialize ZED cameras
    # ------------------------------------------------------------------
    cameras = init_cameras(
        serial_numbers=config.CAMERA_SERIAL_NUMBERS,
        fps=config.CAMERA_FPS,
    )

    # ------------------------------------------------------------------
    # 2. Collect intrinsics from cameras
    # ------------------------------------------------------------------
    intrinsics_list, disto_list = [], []
    for cam in cameras:
        cam_info = cam.get_camera_information()
        intr = cam_info.camera_configuration.calibration_parameters.left_cam
        K = np.array([[intr.fx, 0, intr.cx],
                      [0, intr.fy, intr.cy],
                      [0, 0, 1]], 
                      dtype=np.float64)
        intrinsics_list.append(K)
        disto_list.append(np.array(intr.disto, dtype=np.float64))

    # ------------------------------------------------------------------
    # 3. Define 3D checkerboard points (The fixed common world origin)
    # ------------------------------------------------------------------
    objp = generate_3d_checkerboard_points(config.CALIBRATION_BOARD_SIZE, config.SQUARE_SIZE)

    # ------------------------------------------------------------------
    # 4. Grab frames and convert to grayscale
    # ------------------------------------------------------------------
    image_mats = [sl.Mat() for _ in cameras]
    
    frames = None
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        for _ in range(50):
            out_batch = [None] * len(cameras)
            success = grab_frames(cameras, image_mats, out_batch, executor)
            if success and not any(f is None for f in out_batch):
                frames = list(out_batch)
                break
                
    if frames is None:
        print("\033[91m[ERROR] Failed to grab frames synchronously from all cameras.\033[0m")
        close_cameras(cameras)
        return

    frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    if args.display:
        for i, frame_gray in enumerate(frames_gray):
            cv2.imshow(f"Camera {i} Grayscale", frame_gray)
        print("[INFO] Press any key on the image windows to start corner detection...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------
    # 5. Detect 2D image points for each camera
    # -------------------------------------------------------------------
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    rets, corners_list = zip(*[cv2.findChessboardCorners(f, config.CALIBRATION_BOARD_SIZE, flags=flags) for f in frames_gray])

    if rets.count(True) == len(cameras):
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        
        # -------------------------------------------------------------------
        # 6. Refine corners
        # -------------------------------------------------------------------
        corners_refined = [cv2.cornerSubPix(f, c, (11, 11), (-1, -1), criteria) for f, c in zip(frames_gray, corners_list)]

        # --- Visualize Valid Results ---
        if args.display:
            os.makedirs("snapshot", exist_ok=True)
            for i, (frame, corners, serial) in enumerate(zip(frames, corners_refined, config.CAMERA_SERIAL_NUMBERS)):
                cv2.drawChessboardCorners(frame, config.CALIBRATION_BOARD_SIZE, corners, True)
                cv2.imshow(f"Camera {i} Calibration", frame)
                cv2.imwrite(f"snapshot/calibration_cam_{serial}.jpg", frame)
                
            print("[INFO] Press any key on the image windows to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # -------------------------------------------------------------------
        # 7. Compute extrinsics (R, t) for each camera relative to the WORLD origin
        # -------------------------------------------------------------------
        world_to_cam_results = {}
        cam_in_world_results = {}

        for i, (corners, K, d, serial) in enumerate(zip(corners_refined, intrinsics_list, disto_list, config.CAMERA_SERIAL_NUMBERS)):
            # solvePnP calculates the transformation from the 3D object frame (world) to the camera frame
            success, rvec, tvec = cv2.solvePnP(objp, corners, K, d)
            
            if success:
                R, _ = cv2.Rodrigues(rvec)
                
                # 1. World-to-Camera Transform (Standard Extrinsics for Projection)
                world_to_cam_results[f"camera_{serial}"] = {
                    "R": R.tolist(),
                    "t": tvec.tolist()
                }
                
                # 2. Camera Pose in World Frame (Physical location relative to the mat)
                R_world = R.T
                t_world = -R_world @ tvec
                
                cam_in_world_results[f"camera_{serial}"] = {
                    "R": R_world.tolist(),
                    "t": t_world.tolist()
                }

        # -------------------------------------------------------------------
        # 8. Save results and print summary
        # -------------------------------------------------------------------
        os.makedirs(args.out_dir, exist_ok=True)
        
        save_path_ext = os.path.join(args.out_dir, "world_to_camera_extrinsics.json")
        save_path_pose = os.path.join(args.out_dir, "camera_poses_in_world.json")
        
        with open(save_path_ext, "w") as f:
            json.dump(world_to_cam_results, f, indent=4)
            
        with open(save_path_pose, "w") as f:
            json.dump(cam_in_world_results, f, indent=4)

        print('\n[INFO] ---- Calibration Results (Common World Origin) ----')
        for serial in config.CAMERA_SERIAL_NUMBERS:
            print(f"\n[INFO] Camera {serial} Physical Pose in World Frame:")
            print("R:\n", np.array(cam_in_world_results[f"camera_{serial}"]["R"]))
            print("t:\n", np.array(cam_in_world_results[f"camera_{serial}"]["t"]))

        print('\n[INFO] Calibration files saved to:', args.out_dir)
        print("[INFO] Calibration successful! Extrinsics are locked to the checkerboard origin.")
        close_cameras(cameras)
        
    else:
        if args.display:
            # --- Visualize Failed Results ---
            for i, (frame, ret, corners) in enumerate(zip(frames, rets, corners_list)):
                if ret:
                    cv2.drawChessboardCorners(frame, config.CALIBRATION_BOARD_SIZE, corners, ret)
                cv2.imshow(f"Camera {i} Calibration [FAILED]", frame)
                
        print("\033[91m[ERROR] Checkerboard not detected in all cameras. Please adjust the board and try again.\033[0m")
        print("[INFO] Press any key on the image windows to close and exit...")
        if args.display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        close_cameras(cameras)
        return

if __name__ == "__main__":
    main()