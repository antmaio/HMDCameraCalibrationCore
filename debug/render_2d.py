"""
render_2d.py
Debug script to render 2D keypoints on a single camera stream in real time.
"""

import sys
import os
import cv2
import time
import argparse
import numpy as np
import pyzed.sl as sl

# Try to import optional profiling modules
try:
    import psutil
    import torch
    HAS_PROFILING_LIBS = True
except ImportError:
    HAS_PROFILING_LIBS = False

# Add the root structure to system path to import modules normally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
import json
import threading
import glob
from pythonosc import dispatcher, osc_server
from concurrent.futures import ThreadPoolExecutor
from src.camera import init_cameras, grab_frames, close_cameras
from src.pose import load_model, estimate_poses

# ── OSC & Math helpers ──
class DevicePose:
    def __init__(self):
        self._lock     = threading.Lock()
        self._position = np.zeros(3, dtype=np.float32)
        self._rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        self._updated  = False
 
    def set_position(self, x: float, y: float, z: float) -> None:
        with self._lock:
            self._position[:] = (x, y, z)
            self._updated = True
 
    def set_rotation(self, x: float, y: float, z: float, w: float) -> None:
        with self._lock:
            self._rotation[:] = (x, y, z, w)
 
    def get(self):
        with self._lock:
            updated        = self._updated
            self._updated  = False
            return self._position.copy(), self._rotation.copy(), updated

hmd_pose = DevicePose()
left_pose = DevicePose()
right_pose = DevicePose()

def _osc_position_handler(device_pose: DevicePose, address, x: float, y: float, z: float) -> None:
    device_pose.set_position(x, y, z) 

def _osc_rotation_handler(device_pose: DevicePose, address, x: float, y: float, z: float, w: float) -> None:
    device_pose.set_rotation(x, y, z, w)

def _osc_any_handler(address, *args):
    pass

def start_osc_server(ip: str = "0.0.0.0", port: int = 9000) -> None:
    d = dispatcher.Dispatcher()
    d.map("/hmd/position", lambda addr, x, y, z: _osc_position_handler(hmd_pose, addr, x, y, z))
    d.map("/hmd/rotation", lambda addr, x, y, z, w: _osc_rotation_handler(hmd_pose, addr, x, y, z, w))
    d.map("/left/position", lambda addr, x, y, z: _osc_position_handler(left_pose, addr, x, y, z))
    d.map("/left/rotation", lambda addr, x, y, z, w: _osc_rotation_handler(left_pose, addr, x, y, z, w))
    d.map("/right/position", lambda addr, x, y, z: _osc_position_handler(right_pose, addr, x, y, z))
    d.map("/right/rotation", lambda addr, x, y, z, w: _osc_rotation_handler(right_pose, addr, x, y, z, w))
    d.set_default_handler(_osc_any_handler)
    server = osc_server.ThreadingOSCUDPServer((ip, port), d)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[OSC] Listening for HMD pose on {ip}:{port}")

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

def unity_to_cv(pos, rot_q):
    pos_cv = np.array([pos[0], -pos[1], pos[2]], dtype=np.float32)
    R_u = quaternion_to_rotation_matrix(rot_q)
    M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    R_cv = M @ R_u @ M
    return pos_cv, R_cv

T_c2w_cache = None
def get_cv_to_world_transform(mode):
    global T_c2w_cache
    if T_c2w_cache is not None:
        return T_c2w_cache
        
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    xr_camera = ext.get("xr_camera")
    if xr_camera is None:
        print("[ERROR] Extrinsics for xr_camera not found in JSON.")
        sys.exit(1)
        
    R_xr_world = np.array(xr_camera['R'])
    t_xr_world = np.array(xr_camera['t']).reshape(3)
    
    snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
    json_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.json"))
    if not json_files:
        print("[ERROR] No Snapshot json found.")
        sys.exit(1)
        
    latest_snapshot = sorted(json_files)[-1]
    with open(latest_snapshot, 'r') as f:
        snap = json.load(f)
        
    pos = snap["xrCamera_position"]
    rot = snap["xrCamera_rotation"]
    pos_xr_unity = np.array([pos["x"], pos["y"], pos["z"]])
    rot_xr_unity_q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
    
    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr_unity, rot_xr_unity_q)
    
    R_c2w = R_xr_world @ R_xr_cv.T
    t_c2w = t_xr_world - R_c2w @ pos_xr_cv
    
    T_c2w_cache = (R_c2w, t_c2w)
    return R_c2w, t_c2w

def map_pose_to_world(pos_unity, quat_unity, mode):
    pos_cv, R_cv = unity_to_cv(pos_unity, quat_unity)
    R_c2w, t_c2w = get_cv_to_world_transform(mode)
    
    pos_world = R_c2w @ pos_cv + t_c2w
    R_world = R_c2w @ R_cv
    return pos_world, R_world

def get_projection_matrices(cameras):
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', 'world_to_camera_extrinsics.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    
    proj_matrices = []
    
    for cam, sn in zip(cameras, config.CAMERA_SERIAL_NUMBERS):
        cam_info = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array([[cam_info.fx, 0, cam_info.cx],
                       [0, cam_info.fy, cam_info.cy],
                       [0, 0, 1]])
        
        cam_data = ext.get(f"camera_{sn}")
        if cam_data is None:
            print(f"[ERROR] Extrinsics for camera_{sn} not found in JSON.")
            sys.exit(1)
            
        R = np.array(cam_data['R'])
        t = np.array(cam_data['t']).reshape(3, 1)
        
        P = K @ np.hstack([R, t])
        proj_matrices.append(P)

    return proj_matrices

def get_hmd_points_in_world(mode, axis_len=0.15):
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    with open(filepath, 'r') as f:
        ext = json.load(f)
        
    pts = {}
    for key, data in ext.items():
        R = np.array(data['R'])
        t = np.array(data['t']).reshape(3)
        origin = t
        x_end = origin + R @ np.array([axis_len, 0, 0])
        y_end = origin + R @ np.array([0, axis_len, 0])
        z_end = origin + R @ np.array([0, 0, axis_len])
        pts[key] = {
            'origin': np.array([*origin, 1.0]),
            'x_end': np.array([*x_end, 1.0]),
            'y_end': np.array([*y_end, 1.0]),
            'z_end': np.array([*z_end, 1.0]),
        }
    return pts

def project_point(P, pt_4d):
    uvw = P @ pt_4d
    z = uvw[2]
    if z <= 0.0:
        return None
    return (int(uvw[0]/z), int(uvw[1]/z))

def main() -> None:
    parser = argparse.ArgumentParser(description="Render 2D keypoints on a single camera stream.")
    
    #TODO remove data device transfer from profiling. 
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors"], help="Calibration mode: barycenter or anchors")
    parser.add_argument("--profile", action="store_true", help="Measure inference time and memory usage.")
    parser.add_argument("--save", action="store_true", help="Save displayed frames to disk.")
    parser.add_argument("--show_hmd_calibration", action="store_true", help="Project the calibration 3D point of HMD.")
    parser.add_argument("--show_hmd_motion", action="store_true", help="Project the 3D points of HMD data from Unity OSC.")
    parser.add_argument("--osc-ip", default="0.0.0.0", help="IP to bind the OSC server (default: 0.0.0.0)")
    parser.add_argument("--osc-port", default=9000, type=int, help="UDP port for OSC HMD pose (default: 9000)")
    args = parser.parse_args()

    # Require a calibration mode when projecting HMD calibration or motion to world
    if (args.show_hmd_calibration or args.show_hmd_motion) and args.mode is None:
        print("[ERROR] --mode is required when using --show_hmd_calibration or --show_hmd_motion. Use --mode barycenter or --mode anchors.")
        sys.exit(1)

    if args.show_hmd_motion:
        start_osc_server(args.osc_ip, args.osc_port)

    sn_list = config.CAMERA_SERIAL_NUMBERS
    print(f"[DEBUG] Initializing cameras {sn_list} for 2D visualization...")
    cameras = init_cameras(
        serial_numbers=sn_list,
        fps=config.CAMERA_FPS
    )
    
    if not cameras:
        print("[ERROR] Failed to start cameras.")
        sys.exit(1)
        
    image_mats = [sl.Mat() for _ in cameras]
    proj_matrices = get_projection_matrices(cameras)
    if args.show_hmd_calibration:
        hmd_world_pts = get_hmd_points_in_world(args.mode)
    
    print("[DEBUG] Loading YOLO model...")
    model = load_model(config.YOLO_WEIGHTS, config.YOLO_FORMAT)
    
    window_names = []
    save_dirs = []
    
    # Setup save directory with timestamp if requested
    session_save_root = None
    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_save_root = os.path.join("frames", timestamp, "cameras")
        os.makedirs(session_save_root, exist_ok=True)
        print(f"[DEBUG] Saving frames to {session_save_root}")

    for sn in sn_list:
        win_name = f"Camera {sn} - 2D Keypoints"
        window_names.append(win_name)
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        
        if args.save:
            save_dir = os.path.join(session_save_root, str(sn))
            os.makedirs(save_dir, exist_ok=True)
            save_dirs.append(save_dir)
            print(f"[DEBUG] Created save folder for camera {sn}: {save_dir}")
    
    print("[DEBUG] Stream running. Press ESC to stop.")
    
    # Storage for profiling data
    prof_inf_times = []
    prof_ram_usages = []
    prof_vram_allocs = []
    prof_vram_res = []
    frame_count = 0
    warmup_frames = 100

    # Ensure batched resources for multithreaded capture and batch inference
    res_info = cameras[0].get_camera_information().camera_configuration.resolution
    h_res, w_res = res_info.height, res_info.width
    
    if config.USE_ONNX:
        frames_batch = [np.zeros((h_res, w_res, 3), dtype=np.uint8) for _ in cameras]
    elif HAS_PROFILING_LIBS and torch.cuda.is_available():
        frames_batch = torch.zeros((len(cameras), h_res, w_res, 3), dtype=torch.uint8, device='cuda')
    else:
        frames_batch = np.zeros((len(cameras), h_res, w_res, 3), dtype=np.uint8)
        
    executor = ThreadPoolExecutor(max_workers=len(cameras))

    try:
        while True:
            success = grab_frames(cameras, image_mats, frames_batch, executor)
            
            if not success:
                continue
            
            frame_count += 1
            
            if HAS_PROFILING_LIBS and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            kpts_list = estimate_poses(model, frames_batch)
            
            if HAS_PROFILING_LIBS and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if args.profile and frame_count > warmup_frames:
                # Accumulate metrics
                inf_time_ms = (t1 - t0) * 1000.0
                prof_inf_times.append(inf_time_ms)
                
                if HAS_PROFILING_LIBS:
                    process = psutil.Process(os.getpid())
                    prof_ram_usages.append(process.memory_info().rss / (1024 ** 2))
                    
                    if torch.cuda.is_available():
                        prof_vram_allocs.append(torch.cuda.memory_allocated() / (1024 ** 2))
                        prof_vram_res.append(torch.cuda.memory_reserved() / (1024 ** 2))
                
            for i, frame in enumerate(frames_batch):
                # Copy to numpy if it's a torch tensor so OpenCV can draw
                if hasattr(frame, 'cpu'):
                    frame_2d = frame.cpu().numpy().copy()
                else:
                    frame_2d = frame.copy()
                kpts_2d = kpts_list[i]
                
                # Draw simple points for each detected keypoint
                for pt in kpts_2d:
                    x, y = int(pt[0]), int(pt[1])
                    if x > 0 and y > 0:  # Check for valid keypoints
                        cv2.circle(frame_2d, (x, y), 5, (0, 255, 0), -1)
                        
                # Draw HMD 3D points projected to 2D
                P = proj_matrices[i]
                if args.show_hmd_calibration:
                    for hmd_key, pts_3d in hmd_world_pts.items():
                        origin_2d = project_point(P, pts_3d['origin'])
                        x_2d = project_point(P, pts_3d['x_end'])
                        y_2d = project_point(P, pts_3d['y_end'])
                        z_2d = project_point(P, pts_3d['z_end'])
                        
                        if origin_2d:
                            cv2.circle(frame_2d, origin_2d, 5, (255, 255, 255), -1)
                            cv2.putText(frame_2d, hmd_key, (origin_2d[0]+10, origin_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            if x_2d: cv2.line(frame_2d, origin_2d, x_2d, (0, 0, 255), 2)  # X axis red
                            if y_2d: cv2.line(frame_2d, origin_2d, y_2d, (0, 255, 0), 2)  # Y axis green
                            if z_2d: cv2.line(frame_2d, origin_2d, z_2d, (255, 0, 0), 2)  # Z axis blue

                if args.show_hmd_motion:
                    def draw_device(device_pose, name, color, color_text=None):
                        pos_unity, quat_unity, _ = device_pose.get()
                        pos_world, R_world = map_pose_to_world(pos_unity, quat_unity, args.mode)
                        
                        axis_len = 0.15
                        origin = pos_world
                        x_end = origin + R_world @ np.array([axis_len, 0, 0])
                        y_end = origin + R_world @ np.array([0, axis_len, 0])
                        z_end = origin + R_world @ np.array([0, 0, axis_len])
                        
                        origin_2d = project_point(P, np.array([*origin, 1.0]))
                        x_2d = project_point(P, np.array([*x_end, 1.0]))
                        y_2d = project_point(P, np.array([*y_end, 1.0]))
                        z_2d = project_point(P, np.array([*z_end, 1.0]))
                        
                        if origin_2d:
                            cv2.circle(frame_2d, origin_2d, 5, color, -1)
                            cv2.putText(frame_2d, name, (origin_2d[0]+10, origin_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text or color, 1)
                            if x_2d: cv2.line(frame_2d, origin_2d, x_2d, (0, 0, 255), 3)
                            if y_2d: cv2.line(frame_2d, origin_2d, y_2d, (0, 255, 0), 3)
                            if z_2d: cv2.line(frame_2d, origin_2d, z_2d, (255, 0, 0), 3)

                    draw_device(hmd_pose, "HMD (Motion)", (255, 0, 255))
                    draw_device(left_pose, "Left", (0, 255, 255)) # Yellow in BGR
                    draw_device(right_pose, "Right", (0, 165, 255)) # Orange in BGR

                cv2.imshow(window_names[i], frame_2d)
                
                if args.save:
                    save_path = os.path.join(save_dirs[i], f"frame{frame_count-1}.png")
                    cv2.imwrite(save_path, frame_2d)
            
            # Process GUI events and wait for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                print("[DEBUG] ESC key pressed. Stopping...")
                break
                
    except KeyboardInterrupt:
        print("[DEBUG] Shutting down from KeyboardInterrupt...")

    except Exception as e:
        import traceback
        print(f"[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        executor.shutdown(wait=False)
        # Generate summary report if profiling was enabled
        if args.profile and len(prof_inf_times) > 0:
            print("\n" + "="*45)
            print("               PROFILING REPORT")
            print("="*45)
            print(f"Total Frames analyzed : {len(prof_inf_times)}")
            print(f"Average Infer time    : {np.mean(prof_inf_times):.2f} ± {np.std(prof_inf_times):.2f} ms")
            
            if HAS_PROFILING_LIBS and len(prof_ram_usages) > 0:
                print(f"System RAM (RSS)      : {np.mean(prof_ram_usages):.1f} ± {np.std(prof_ram_usages):.1f} MB")
                
                if len(prof_vram_allocs) > 0:
                    print(f"GPU VRAM Allocated    : {np.mean(prof_vram_allocs):.1f} ± {np.std(prof_vram_allocs):.1f} MB")
                    print(f"GPU VRAM Reserved     : {np.mean(prof_vram_res):.1f} ± {np.std(prof_vram_res):.1f} MB")
            else:
                print("Skipped RAM/VRAM profiling. Check 'psutil' / 'torch'.")
            print("="*45 + "\n")
            
        cv2.destroyAllWindows()
        close_cameras(cameras)

if __name__ == "__main__":
    main()
