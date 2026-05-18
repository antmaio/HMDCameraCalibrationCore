"""
send_yolo_positions.py

Main entry point for streaming triangulated YOLOv8 pose keypoints from ZED cameras to Unity via OSC.
"""

import sys
import os
import time
import cv2
import json
import glob
import argparse
import numpy as np
import pyzed.sl as sl
from concurrent.futures import ThreadPoolExecutor

# Ensure src/ is in the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from pose import load_model, estimate_poses
from camera import init_cameras, grab_frames, close_cameras
from triangulation import triangulate_multiview, build_projection_matrix, _dlt_triangulate_point
from osc_sender import OscSenderSleepBasedRateLimiter
from utils import build_intrinsics_from_zed, cv_to_unity, load_latest_snapshot_json, unity_to_cv

def get_world_to_unity_transform(mode: str):
    """Calculates transforming OpenCV World to Unity coordinates from calibration files."""
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    
    try:
        with open(extrinsics_path, 'r') as f:
            ext = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Extrinsics {extrinsics_path} not found.")
        sys.exit(1)
        
    xr_camera = ext.get("xr_camera")
    if xr_camera is None:
        print("[ERROR] Extrinsics for xr_camera not found in JSON.")
        sys.exit(1)
        
    R_xr_world = np.array(xr_camera['R'])
    t_xr_world = np.array(xr_camera['t']).reshape(3)
    
    # Load snapshot to get Unity pose at calibration
    snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
    snap = load_latest_snapshot_json(snapshot_dir)

    space_map = {
        "barycenter": "guardianSpace",
        "anchors": "anchorSpace"
    }
    space_key = space_map.get(mode, "trackingSpace")
    space_data = snap.get(space_key, snap.get("trackingSpace", snap))

    pos = space_data.get("xrCamera_position")
    rot = space_data.get("xrCamera_rotation")
    if pos is None or rot is None:
        if "xrCamera_position" in snap and "xrCamera_rotation" in snap:
            pos = snap["xrCamera_position"]
            rot = snap["xrCamera_rotation"]
        else:
            available = ", ".join(space_data.keys())
            print(f"[ERROR] Snapshot JSON does not contain xrCamera_position/xrCamera_rotation in '{space_key}'. Available keys: {available}")
            sys.exit(1)

    pos_xr_unity = np.array([pos["x"], pos["y"], pos["z"]])
    rot_xr_unity_q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])

    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr_unity, rot_xr_unity_q)
    
    R_c2w = R_xr_world @ R_xr_cv.T
    t_c2w = t_xr_world - R_c2w @ pos_xr_cv
    
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    return R_w2c, t_w2c

def get_projection_matrices(cameras: list) -> list:
    """Build projection matrices P = K @ [R | t] for all cameras."""
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', 'world_to_camera_extrinsics.json'))
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

# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Start the YOLO pose estimation loop and stream 3D points over OSC to Unity."""
    parser = argparse.ArgumentParser(description="Stream YOLO poses to Unity via OSC.")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors"], required=True,
                        help="Calibration mode: barycenter or anchors")
    parser.add_argument("--display", action="store_true", help="Display OpenCV camera frames")
    args = parser.parse_args()

    print("=== YOLO Pose to Unity OSC Streamer ===")
    print(f"Mode: {args.mode}, Display: {args.display}")

    cam_serials = config.CAMERA_SERIAL_NUMBERS

    # Init Cameras
    cameras = init_cameras(cam_serials, resolution=sl.RESOLUTION.HD720, fps=config.CAMERA_FPS)
    if len(cameras) < 2:
        print("[ERROR] Need at least 2 cameras for triangulation.")
        close_cameras(cameras)
        return

    # Extract projection matrices mapping World to Camera
    try:
        proj_matrices = get_projection_matrices(cameras)
        print(f"[CALIB] Loaded {len(proj_matrices)} projection matrices.")
    except Exception as e:
        print(f"[ERROR] Loading projection matrices: {e}")
        close_cameras(cameras)
        return

    import torch

    # Pre-allocate batched frames and image mats
    res = cameras[0].get_camera_information().camera_configuration.resolution
    h, w = res.height, res.width
    
    if torch.cuda.is_available():
        frames_batch = torch.zeros((len(cameras), h, w, 3), dtype=torch.uint8, device='cuda')
    else:
        frames_batch = np.zeros((len(cameras), h, w, 3), dtype=np.uint8)
        
    mats = [sl.Mat() for _ in cameras]
    executor = ThreadPoolExecutor(max_workers=len(cameras))

    # Load YOLO
    yolo_model = load_model(config.YOLO_WEIGHTS, config.USE_ONNX)

    # Init OSC Sender
    sender = OscSenderSleepBasedRateLimiter(ip=config.HMD_OSC_IP, port=config.SEND_OSC_PORT, target_hz=60.0)
    sender.start()
    
    # Load world to Unity transform
    try: 
        R_w2c, t_w2c = get_world_to_unity_transform(args.mode)
    except Exception as e:
         print(f"[ERROR] Loading world transform: {e}")
         sender.stop()
         close_cameras(cameras)
         return

    print("\n[STREAM] Running... Press Ctrl+C in terminal (or 'q' in CV window if --display) to exit.")

    # ── Send-rate measurement ─────────────────────────────────────────
    _rate_counter   = 0
    _rate_window    = 2.0          # print every N seconds
    _rate_last_time = time.perf_counter()

    try:
        while True:
            # Grab BGR frames from all cameras concurrently into pre-allocated numpy array
            success = grab_frames(cameras, mats, frames_batch, executor)
            if not success:
                continue  # Skip frame if a camera dropped

            # Estimate 2D poses -> List of (17, 2) per camera
            poses_2d = estimate_poses(yolo_model, frames_batch)

            # Triangulate to 3D.  Poses_2d holds a list of (17,2) for N cameras.
            if len(proj_matrices) == len(cameras):
                kpts_3d = []
                for kpt_idx in range(17):
                    # Gather this keypoint from all cameras
                    cams_kpt = [poses_2d[cam_idx][kpt_idx] for cam_idx in range(len(cameras))]
                    
                    # Only triangulate if they are valid (non-zero) - basic confidence check.
                    valid_views = [i for i, pt in enumerate(cams_kpt) if pt[0] != 0 and pt[1] != 0]
                    
                    if len(valid_views) >= 2:
                        # Extract valid points and corresponding projection matrices
                        valid_pts = [cams_kpt[i] for i in valid_views]
                        valid_projs = [proj_matrices[i] for i in valid_views]
                        pk = _dlt_triangulate_point(valid_projs, valid_pts)
                        
                        # Apply OpenCV world to camera world -> unity transform
                        corner_cv = R_w2c @ pk + t_w2c
                        corner_unity = cv_to_unity(corner_cv)
                        kpts_3d.append(corner_unity)
                    else:
                        kpts_3d.append(np.zeros(3))
                        
                kpts_3d_unity = np.array(kpts_3d, dtype=np.float32)
                sender.update(kpts_3d_unity)

            # --- Rate measurement ---
            _rate_counter += 1
            t_now = time.perf_counter()
            if t_now - _rate_last_time >= _rate_window:
                fps = _rate_counter / (t_now - _rate_last_time)
                print(f"[STREAM] Processing Rate: {fps:.2f} Hz")
                _rate_counter = 0
                _rate_last_time = t_now

            # --- Optional debug visualization ---
            if args.display:
                for i in range(len(cameras)):
                    frame_source = frames_batch[i]
                    if hasattr(frame_source, 'cpu'):
                        frame = frame_source.cpu().numpy().copy()
                    else:
                        frame = frame_source.copy()
                        
                    kpts = poses_2d[i]
                    for x, y in kpts:
                        if x != 0 and y != 0:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    cv2.imshow(f"Cam {i}", cv2.resize(frame, (640, 360)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        executor.shutdown(wait=False)
        sender.stop()
        close_cameras(cameras)
        if args.display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()