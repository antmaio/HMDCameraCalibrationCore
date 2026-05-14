"""
render_3d.py
Debug script to render 3D keypoints from stereo 2D triangulation using a single ZED camera.
Additionally receives HMD pose (position + rotation) via OSC and renders it in the 3D scene.
 
OSC messages expected:
    /hmd/position   float x  float y  float z         (metres, Unity left-handed space)
    /hmd/rotation   float x  float y  float z  float w (quaternion, Unity space)
 
Install OSC dependency:
    pip install python-osc
"""
 

import sys
import os
import cv2
import time
import argparse
import json
import threading
import numpy as np
import pyzed.sl as sl
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pythonosc import dispatcher, osc_server
from concurrent.futures import ThreadPoolExecutor

# Add the root and src structure to system path to import modules normally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config 
from src.pose import load_model, estimate_poses
from src.triangulation import triangulate_multiview

# ── OSC shared state (written by OSC thread, read by Qt thread) ──────────────
class DevicePose:
    """Thread-safe container for the latest device pose received over OSC."""
 
    def __init__(self):
        self._lock     = threading.Lock()
        self._position = np.zeros(3, dtype=np.float32)   # x y z (Unity space)
        self._rotation = np.array([0, 0, 0, 1], dtype=np.float32)  # xyzw quaternion
        self._updated  = False
 
    def set_position(self, x: float, y: float, z: float) -> None:
        with self._lock:
            self._position[:] = (x, y, z)
            self._updated = True
 
    def set_rotation(self, x: float, y: float, z: float, w: float) -> None:
        with self._lock:
            self._rotation[:] = (x, y, z, w)
 
    def get(self):
        """Returns (position np.array(3,), rotation np.array(4,), updated bool)."""
        with self._lock:
            updated        = self._updated
            self._updated  = False
            return self._position.copy(), self._rotation.copy(), updated

    
hmd_pose = DevicePose()
left_pose = DevicePose()
right_pose = DevicePose()

# ── OSC server (runs in background thread) ───────────────────────────────────
def _osc_position_handler(device_pose: DevicePose, address, x: float, y: float, z: float) -> None:
    # print(f"[OSC-RECV] Position: {x:.3f}, {y:.3f}, {z:.3f}")
    device_pose.set_position(x, y, z) 
def _osc_rotation_handler(device_pose: DevicePose, address, x: float, y: float, z: float, w: float) -> None:
    # print(f"[OSC-RECV] Rotation: {x:.3f}, {y:.3f}, {z:.3f}, {w:.3f}")
    device_pose.set_rotation(x, y, z, w)
def _osc_any_handler(address, *args):
    pass
    # print(f"[OSC-DEBUG] Received message on {address}: {args}")

def start_osc_server(ip: str = "0.0.0.0", port: int = 9000) -> None:
    """Starts the OSC UDP server in a daemon thread."""
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

# ── Coordinate helpers ────────────────────────────────────────────────────────
def OpenCV2PyQtGraphAxisSystem(pts: np.ndarray) -> np.ndarray:
    """
    Apply +180 deg rotation around X axis to convert OpenCV 
    (X: right, Y: down, Z: forward) to a Z-up coordinate system 
    (X: right, Y: up, Z: backward).
    Rotation matrix around X by 180 deg:
    [[1,  0,  0],
     [0, -1,  0],
     [0,  0, -1]]
    """
    out = pts.copy()
    if out.ndim == 1:
        # Single point [x, y, z] -> [x, -y, -z]
        out[1] = -out[1]
        out[2] = -out[2]
    else:
        # Array of points [[x, y, z], ...] -> [[x, -y, -z], ...]
        out[:, 1] = -out[:, 1]
        out[:, 2] = -out[:, 2]
    return out

def Unity2PyQtGraphAxisSystem(pos: np.ndarray) -> np.ndarray:
    return OpenCV2PyQtGraphAxisSystem(pos)

def unity_to_cv(pos, rot_q):
    """
    Map Unity's tracking coordinate space (LH: x-rt, y-up, z-fwd) 
    to OpenCV coordinate space (RH: x-rt, y-dn, z-fwd).
    """
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
        
    # Load extrinsics from JSON
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    xr_camera = ext.get("xr_camera")
    if xr_camera is None:
        print("[ERROR] Extrinsics for xr_camera not found in JSON.")
        sys.exit(1)
        
    R_xr_world = np.array(xr_camera['R'])
    t_xr_world = np.array(xr_camera['t']).reshape(3)
    
    # Load snapshot to get Unity pose at calibration
    import glob
    snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
    json_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.json"))
    if not json_files:
        print("[ERROR] No Snapshot json found.")
        sys.exit(1)
        
    latest_snapshot = sorted(json_files)[-1]
    with open(latest_snapshot, 'r') as f:
        snap = json.load(f)
        
    # Get calibration xrCamera pose in CV space
    pos = snap["xrCamera_position"]
    rot = snap["xrCamera_rotation"]
    pos_xr_unity = np.array([pos["x"], pos["y"], pos["z"]])
    rot_xr_unity_q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
    
    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr_unity, rot_xr_unity_q)
    
    # T_world = T_cv_to_world * T_cv
    # R_world = R_c2w * R_cv => R_c2w = R_world * R_cv.T
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

    
        
def get_extrinsic_matrices_hmd():
    # Load extrinsics from JSON
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', 'world_to_hmd_extrinsics.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    xr_camera = ext.get("xr_camera")
    if xr_camera is None:
        print(f"[ERROR] Extrinsics for xr_camera not found in JSON.")
        sys.exit(1)
    R = np.array(xr_camera['R'])
    t = np.array(xr_camera['t']).reshape(3, 1)
    P = np.hstack([R, t])
    return P


def make_hmd_axes_pts(pos_world: np.ndarray, R_world: np.ndarray,
                      axis_len: float = 0.15) -> np.ndarray:
    """
    Build the 6-point array (3 axis segments) for a GLLinePlotItem
    representing the HMD frame, expressed in PyQtGraph space.
    """
    origin = pos_world
    x_end  = origin + R_world @ np.array([axis_len, 0, 0], dtype=np.float32)
    y_end  = origin + R_world @ np.array([0, axis_len, 0], dtype=np.float32)
    z_end  = origin + R_world @ np.array([0, 0, axis_len], dtype=np.float32)
 
    pts = np.array([origin, x_end,
                    origin, y_end,
                    origin, z_end], dtype=np.float32)
 
    # Convert each point to PyQtGraph space
    return OpenCV2PyQtGraphAxisSystem(pts)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x y z w] to a 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

# ── Camera / projection helpers  ───────────────────────────────────
def get_projection_matrices(cameras: list[sl.Camera]):
    # Load extrinsics from JSON
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
def set_world_axis(view, axes_colors:np.ndarray)-> None:

    #Draw World (origin) axes (X=Red, Y=Green, Z=Blue)
    world_axes_pts = np.array([
        [0, 0, 0], [1, 0, 0],  # X
        [0, 0, 0], [0, 1, 0],  # Y
        [0, 0, 0], [0, 0, 1]   # Z
    ], dtype=np.float32) 
    # Convert OpenCV coordinates to PyQtGraph coordinates
    world_axes_pts = OpenCV2PyQtGraphAxisSystem(world_axes_pts)


    world_axis_line = gl.GLLinePlotItem(pos=world_axes_pts, color=axes_colors, mode='lines', width=3)
    view.addItem(world_axis_line)

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:

    parser = argparse.ArgumentParser(description="Render 3D keypoints from multi-camera triangulation.")
    #parser.add_argument("--mode", type=str, choices=["barycenter", "anchors"], help="Calibration mode: barycenter or anchors")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save", action="store_true", help="Save displayed frames to disk.")
    parser.add_argument("--osc-ip",   default="0.0.0.0",
                        help="IP to bind the OSC server (default: 0.0.0.0)")
    parser.add_argument("--osc-port", default=9000, type=int,
                        help="UDP port for OSC HMD pose (default: 9000)")
    args = parser.parse_args()

    # Enable OpenGL debug logging
    if args.verbose:   
        os.environ["QT_LOGGING_RULES"] = "qt.qpa.gl=true"

    # ── Start OSC receiver ────────────────────────────────────────────
    start_osc_server(args.osc_ip, args.osc_port)
 
    # ── Cameras ───────────────────────────────────────────────────────
    # Use specified cameras from config
    print(f"[INFO] Initializing cameras {config.CAMERA_SERIAL_NUMBERS} for 3D triangulation...")
    
    from src.camera import init_cameras
    cameras = init_cameras(config.CAMERA_SERIAL_NUMBERS, sl.RESOLUTION.HD720, config.CAMERA_FPS)
    
    if len(cameras) < 2:
        print("[ERROR] Need at least 2 cameras for multi-view triangulation.")
        sys.exit(1)
        
    proj_matrices = get_projection_matrices(cameras)
    
    image_mats = [sl.Mat() for _ in cameras]
    
    import torch
    
    # Ensure batched resources for multithreaded capture and batch inference
    res_info = cameras[0].get_camera_information().camera_configuration.resolution
    h_res, w_res = res_info.height, res_info.width
    
    if torch.cuda.is_available():
        frames_batch = torch.zeros((len(cameras), h_res, w_res, 3), dtype=torch.uint8, device='cuda')
    else:
        frames_batch = np.zeros((len(cameras), h_res, w_res, 3), dtype=np.uint8)
        
    executor = ThreadPoolExecutor(max_workers=len(cameras))
    
    print("[INFO] Loading YOLO model...")
    model = load_model(config.YOLO_WEIGHTS, config.USE_ONNX) 
    
        # ── PyQtGraph scene ───────────────────────────────────────────────
    # Setup PyQtGraph 3D plot
    app = pg.mkQApp("3D Pose Visualization")
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle(f"Multi-Camera 3D Pose")
    view.setCameraPosition(distance=5, elevation=20, azimuth=45)
    
    grid = gl.GLGridItem()
    grid.scale(1, 1, 1) # Scales the grid
    view.addItem(grid)
    
    scatter = gl.GLScatterPlotItem(pos=np.array([[0,0,0]], dtype=np.float32), color=(0, 0.5, 1, 1), size=10)
    scatter.setVisible(False)
    view.addItem(scatter)
    
    #Draw World (origin) axes (X=Red, Y=Green, Z=Blue)
    axes_colors = np.array([
        [1, 0, 0, 1], [1, 0, 0, 1],
        [0, 1, 0, 1], [0, 1, 0, 1],
        [0, 0, 1, 1], [0, 0, 1, 1]
    ], dtype=np.float32)
    set_world_axis(view, axes_colors)
    
    # Load extrinsics to draw camera axes
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', 'world_to_camera_extrinsics.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    
    for sn in config.CAMERA_SERIAL_NUMBERS:
        cam_data = ext.get(f"camera_{sn}")
        if cam_data:
            R = np.array(cam_data['R'])
            t = np.array(cam_data['t']).reshape(3)
            
            # Cam pos and orientation in World frame
            # Reduce axis length (e.g., to 0.2 meters)
            axis_len = 0.4
            pos_cam = -R.T @ t
            x_axis = pos_cam + R.T @ np.array([axis_len, 0, 0])
            y_axis = pos_cam + R.T @ np.array([0, axis_len, 0])
            z_axis = pos_cam + R.T @ np.array([0, 0, axis_len])
            
            cam_axes_pts = np.array([
                pos_cam, x_axis,
                pos_cam, y_axis,
                pos_cam, z_axis
            ], dtype=np.float32)
            # Convert OpenCV coordinates to PyQtGraph coordinates
            cam_axes_pts = OpenCV2PyQtGraphAxisSystem(cam_axes_pts)
            
            cam_axis_line = gl.GLLinePlotItem(pos=cam_axes_pts, color=axes_colors, mode='lines', width=3)
            view.addItem(cam_axis_line)
    
    # We use a single GLLinePlotItem with mode='lines' to draw unconnected segments
    skeleton_line = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]], dtype=np.float32), 
                                      color=(0, 1, 0, 1), width=2, mode='lines')
    skeleton_line.setVisible(False)
    view.addItem(skeleton_line)

    # ── HMD pose display items ────────────────────────────────────────
    # Axes (X=red, Y=green, Z=blue) for HMD orientation
    hmd_axis_line = gl.GLLinePlotItem(
        pos=np.zeros((6, 3), dtype=np.float32),
        color=axes_colors, mode='lines', width=4)
    hmd_axis_line.setVisible(False)
    view.addItem(hmd_axis_line)
 
    # Small scatter dot at HMD origin
    hmd_dot = gl.GLScatterPlotItem(
        pos=np.array([[0, 0, 0]], dtype=np.float32),
        color=(1, 1, 1, 1), size=12)
    hmd_dot.setVisible(False)
    view.addItem(hmd_dot)

    left_dot = gl.GLScatterPlotItem(
        pos=np.array([[0, 0, 0]], dtype=np.float32),
        color=(1, 1, 0, 1), size=15) # Yellow
    left_dot.setVisible(False)
    view.addItem(left_dot)

    right_dot = gl.GLScatterPlotItem(
        pos=np.array([[0, 0, 0]], dtype=np.float32),
        color=(1, 0.5, 0, 1), size=15) # Orange
    right_dot.setVisible(False)
    view.addItem(right_dot)

    print("[INFO] Stream running. Close window to stop.")
    
    save_dir = None
    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("frames", timestamp, "3d")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[DEBUG] Saving 3D frames to {save_dir}")

    frame_count = 0
    
    # ── Per-frame update ──────────────────────────────────────────────
    def update():
        nonlocal frame_count
        from src.camera import grab_frames

        # ── Update device poses from OSC (every frame, independent of cameras) ──
        def update_device_graphics(device_pose, dot_item, line_item=None):
            pos_unity, quat_unity, pose_updated = device_pose.get()
            if pose_updated or dot_item.visible():
                pos_world, R_world = map_pose_to_world(pos_unity, quat_unity)
                origin_pyqt = OpenCV2PyQtGraphAxisSystem(pos_world)

                if line_item is not None:
                    axes_pts = make_hmd_axes_pts(pos_world, R_world, axis_len=0.15)
                    line_item.setData(pos=axes_pts)
                    line_item.setVisible(True)
     
                dot_item.setData(pos=origin_pyqt.reshape(1, 3))
                dot_item.setVisible(True)

        update_device_graphics(hmd_pose, hmd_dot, hmd_axis_line)
        update_device_graphics(left_pose, left_dot)
        update_device_graphics(right_pose, right_dot)
 
        # ── Camera grab + body pose ──────────────────────────────────
        success = grab_frames(cameras, image_mats, frames_batch, executor)
        
        if success:
            frame_count += 1
            # Estimate 2D poses in all views
            kpts_list = estimate_poses(model, frames_batch)
            
            # Triangulate
            pose_3d = triangulate_multiview(
                kpts_list=kpts_list,
                proj_matrices=proj_matrices
            )
            
            valid = ~np.all(pose_3d == 0, axis=1)
            
            if np.any(valid):
                # Convert OpenCV coordinates (X right, Y down, Z forward) 
                # to PyQtGraph coordinates (X right, Y forward, Z up)
                plot_pos = OpenCV2PyQtGraphAxisSystem(pose_3d)
                
                valid_pts = plot_pos[valid]
                try:
                    scatter.setData(pos=valid_pts)
                    scatter.setVisible(True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                
                # Update skeleton lines array
                edge_positions = []
                for edge in config.SKELETON_EDGES:
                    p1, p2 = edge
                    if valid[p1] and valid[p2]:
                        edge_positions.append(plot_pos[p1])
                        edge_positions.append(plot_pos[p2])

                if len(edge_positions) > 0:
                    try:
                        skeleton_line.setData(pos=np.array(edge_positions, dtype=np.float32))
                        skeleton_line.setVisible(True)
                    except Exception: 
                        import traceback
                        traceback.print_exc()
                else:
                    skeleton_line.setVisible(False)
            else:
                scatter.setVisible(False)
                skeleton_line.setVisible(False)
                
            if args.save:
                save_path = os.path.join(save_dir, f"frame{frame_count-1}.png")
                # Wait for the scene to update before grabbing
                app.processEvents()
                img = view.grabFramebuffer()
                img.save(save_path)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1) 

    try:
        pg.exec()
    except KeyboardInterrupt:
        print("[DEBUG] Shutting down...")
    finally:
        timer.stop()
        executor.shutdown(wait=False)
        for cam in cameras:
            cam.close()

if __name__ == "__main__":
    main()