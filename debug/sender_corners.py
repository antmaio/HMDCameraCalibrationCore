"""
sender_corners.py
Finds chessboard corners, triangulates them across multiple ZED cameras,
converts from OpenCV world space to Unity tracking space, and sends via OSC.

Usage:
    python sender_corners.py --mode barycenter [--osc-ip 127.0.0.1] [--osc-port 5005]
"""

import sys
import os
import cv2
import json
import argparse
import numpy as np
import threading
import pyzed.sl as sl
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import dispatcher, osc_server
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from concurrent.futures import ThreadPoolExecutor

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.camera import init_cameras, grab_frames, close_cameras
from src.triangulation import build_projection_matrix, _dlt_triangulate_point

# ──────────────────────────────────────────────────────────────────────────────
# Unity OSC Receiver
# ──────────────────────────────────────────────────────────────────────────────

class CornerData:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        
    def set(self, idx, x, y, z, color):
        with self.lock:
            self.data[idx] = {
                "pos": np.array([x, y, z], dtype=np.float32), 
                "color": color
            }
            
    def get_all(self):
        with self.lock:
            return {k: v.copy() for k, v in self.data.items()}

unity_corners = CornerData()

def _osc_corner_handler(address, x: float, y: float, z: float, color: str):
    try:
        idx = int(address.replace("/checkerboard/corner", ""))
        unity_corners.set(idx, x, y, z, color)
    except Exception as e:
        print(f"[OSC-RECV] Error parsing {address}: {e}")

def start_osc_server(ip: str = "0.0.0.0", port: int = config.LISTEN_OSC_PORT):
    d = dispatcher.Dispatcher()
    d.map("/checkerboard/corner*", _osc_corner_handler)
    server = osc_server.ThreadingOSCUDPServer((ip, port), d)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[OSC] Listening for Unity corners on {ip}:{port}")

def project_point(P, pt_3d):
    pt_4d = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0], dtype=np.float32)
    uvw = P @ pt_4d
    z = uvw[2]
    if z <= 0.0:
        return None
    return (int(uvw[0]/z), int(uvw[1]/z))

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate System Conversion Helpers
# ──────────────────────────────────────────────────────────────────────────────

def OpenCV2PyQtGraphAxisSystem(pts: np.ndarray) -> np.ndarray:
    """
    Apply +180 deg rotation around X axis to convert OpenCV 
    (X: right, Y: down, Z: forward) to a Z-up coordinate system 
    (X: right, Y: up, Z: backward).
    """
    out = pts.copy()
    if out.ndim == 1:
        out[1] = -out[1]
        out[2] = -out[2]
    else:
        out[:, 1] = -out[:, 1]
        out[:, 2] = -out[:, 2]
    return out

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x y z w] to a 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

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

def cv_to_unity(pos_cv):
    """
    Inverse of unity_to_cv for positions.
    Convert from OpenCV coordinates to Unity coordinates.
    """
    # Unity: (x, -y, z) where OpenCV is (x, y, z), though OpenCV y is down.
    # From unity_to_cv: pos_cv = [pos_u[0], -pos_u[1], pos_u[2]]
    # Therefore: pos_u = [pos_cv[0], -pos_cv[1], pos_cv[2]]
    pos_unity = np.array([pos_cv[0], -pos_cv[1], pos_cv[2]], dtype=np.float32)
    return pos_unity

# ──────────────────────────────────────────────────────────────────────────────
# Calibration Transformation Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_world_to_unity_transform(mode: str):
    import glob
    # Same logic as render_3d.py's get_cv_to_world_transform, but inverted.
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    left_camera = ext.get("left_eye")
    if left_camera is None:
        print("[ERROR] Extrinsics for left_eye not found in JSON.")
        sys.exit(1)
        
    R_xr_world = np.array(left_camera['R'])
    t_xr_world = np.array(left_camera['t']).reshape(3)
    
    # Load snapshot to get Unity pose at calibration
    snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
    json_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.json"))
    if not json_files:
        print("[ERROR] No Snapshot json found.")
        sys.exit(1)
        
    latest_snapshot = sorted(json_files)[-1]
    with open(latest_snapshot, 'r') as f:
        snap = json.load(f)
        
    # Snapshots may store positions in the root, or nested inside 'trackingSpace'
    space_data = snap.get("trackingSpace", snap)
    
    pos = space_data["left_camera_position"]
    rot = space_data["left_camera_rotation"]
    pos_xr_unity = np.array([pos["x"], pos["y"], pos["z"]])
    rot_xr_unity_q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
    
    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr_unity, rot_xr_unity_q)
    
    # render_3d computes R_c2w = R_xr_world @ R_xr_cv.T
    R_c2w = R_xr_world @ R_xr_cv.T
    # t_c2w = t_xr_world - R_c2w @ pos_xr_cv
    t_c2w = t_xr_world - R_c2w @ pos_xr_cv
    
    # We want world_to_cv:
    # pos_world = R_c2w @ pos_cv + t_c2w
    # pos_cv = R_c2w^T @ (pos_world - t_c2w) = R_c2w^T @ pos_world - R_c2w^T @ t_c2w
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
        K = np.array([[cam_info.fx, 0, cam_info.cx],
                      [0, cam_info.fy, cam_info.cy],
                      [0, 0, 1]], dtype=np.float32)  
        
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
# Chessboard Detection and Triangulation
# ──────────────────────────────────────────────────────────────────────────────

def find_chessboard_corners(frames: list | np.ndarray, board_size: tuple) -> list:
    corners_all = []
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_all.append(corners_refined.reshape(-1, 2))
            print(f"[Camera {i}] Chessboard found: {corners_refined.shape[0]} corners")
        else:
            corners_all.append(None)
            print(f"[Camera {i}] Chessboard not found")
    return corners_all

def get_extreme_corners(corners_array: np.ndarray, board_size: tuple) -> np.ndarray:
    if corners_array is None or len(corners_array) == 0:
        return None
    w, h = board_size
    # findChessboardCorners returns corners row-by-row, from one side to the other.
    # We mapping them to consistent logical positions:
    idx_tl = 0
    idx_tr = w - 1
    idx_bl = (h - 1) * w
    idx_br = w * h - 1
    
    # Returning in order: [TopLeft, TopRight, BottomLeft, BottomRight]
    return np.array([
        corners_array[idx_tl], # Logical TopLeft
        corners_array[idx_tr], # Logical TopRight
        corners_array[idx_bl], # Logical BottomLeft
        corners_array[idx_br]  # Logical BottomRight
    ], dtype=np.float32)

def triangulate_corners(corners_all: list, proj_matrices: list) -> np.ndarray:
    valid_cameras = [(i, corners, P) for i, (corners, P) in enumerate(zip(corners_all, proj_matrices)) if corners is not None]
    if len(valid_cameras) < 2:
        print("[WARNING] Need at least 2 cameras to triangulate. Got:", len(valid_cameras))
        return None
    
    corners_3d = np.zeros((4, 3), dtype=np.float32)
    for corner_idx in range(4):
        points_2d = [corners[corner_idx] for _, corners, _ in valid_cameras]
        proj_mats = [P for _, _, P in valid_cameras]
        corners_3d[corner_idx] = _dlt_triangulate_point(proj_mats, points_2d)
    return corners_3d

def transform_to_unity(corners_world: np.ndarray, mode: str) -> np.ndarray:
    R_w2c, t_w2c = get_world_to_unity_transform(mode)
    corners_unity = np.zeros_like(corners_world)
    for i, corner_world in enumerate(corners_world):
        corner_cv = R_w2c @ corner_world + t_w2c
        corners_unity[i] = cv_to_unity(corner_cv)
    return corners_unity

# ──────────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Find and triangulate chessboard corners, send via OSC.")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors"], required=True,
                        help="Calibration mode: barycenter or anchors")
    parser.add_argument("--osc-ip", type=str, default=config.HMD_OSC_IP, help="OSC server IP")
    parser.add_argument("--osc-port", type=int, default=config.SEND_OSC_PORT, help="OSC server port")
    args = parser.parse_args()
    
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] OSC target: {args.osc_ip}:{args.osc_port}")
    print(f"[INFO] OSC source: 0.0.0.0:{config.LISTEN_OSC_PORT}")
    print(f"[INFO] Board size: {config.CALIBRATION_BOARD_SIZE}")
    
    osc_client = SimpleUDPClient(args.osc_ip, args.osc_port) #send corners to Unity  
    start_osc_server("0.0.0.0", config.LISTEN_OSC_PORT) #receive corners from Unity for visualization

    print(f"[INFO] Initializing cameras {config.CAMERA_SERIAL_NUMBERS}...")
    cameras = init_cameras(config.CAMERA_SERIAL_NUMBERS, sl.RESOLUTION.HD720, config.CAMERA_FPS)
    if len(cameras) < 2:
        print("[ERROR] Need at least 2 cameras.")
        close_cameras(cameras)
        sys.exit(1)
        
    image_mats = [sl.Mat() for _ in cameras]
    
    import torch
    
    # Ensure batched resources for multithreaded capture
    res_info = cameras[0].get_camera_information().camera_configuration.resolution
    h_res, w_res = res_info.height, res_info.width
    
    if torch.cuda.is_available():
        frames_batch = torch.zeros((len(cameras), h_res, w_res, 3), dtype=torch.uint8, device='cuda')
    else:
        frames_batch = np.zeros((len(cameras), h_res, w_res, 3), dtype=np.uint8)
        
    executor = ThreadPoolExecutor(max_workers=len(cameras))
    
    proj_matrices = get_projection_matrices(cameras)
    
    # Setup 3D Scene
    app = pg.mkQApp("3D Corner Visualization")
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle(f"Extracted Corners (Mode: {args.mode})")
    view.setCameraPosition(distance=3, elevation=20, azimuth=45)
    
    grid = gl.GLGridItem()
    grid.scale(1, 1, 1)
    view.addItem(grid)
    
    # Draw World (origin) axes (X=Red, Y=Green, Z=Blue)
    world_axes_pts = OpenCV2PyQtGraphAxisSystem(np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [0, 0, 0], [0, 0, 1]
    ], dtype=np.float32))
    axes_colors = np.array([
        [1, 0, 0, 1], [1, 0, 0, 1],
        [0, 1, 0, 1], [0, 1, 0, 1],
        [0, 0, 1, 1], [0, 0, 1, 1]
    ], dtype=np.float32)
    world_axis_line = gl.GLLinePlotItem(pos=world_axes_pts, color=axes_colors, mode='lines', width=3)
    view.addItem(world_axis_line)
    
    #RGB format
    scatter_colors = np.array([
        [0, 1, 0, 1], # GREEN Top right
        [1, 1, 0, 1], # YELLOW Bottom right 
        [1, 0, 0, 1], # RED Top left
        [0, 0, 1, 1]  # BLUE Bottom Left
    ], dtype=np.float32)

    scatter = gl.GLScatterPlotItem(pos=np.zeros((4, 3), dtype=np.float32), color=scatter_colors, size=15)
    scatter.setVisible(False)
    view.addItem(scatter)
    
    # Unity-space corners (visualized in same scene for comparison)
    # We use a slightly smaller size and different appearance (e.g. outline or transparency)
    # scatter_unity = gl.GLScatterPlotItem(pos=np.zeros((4, 3), dtype=np.float32), color=scatter_colors, size=10)
    # scatter_unity.setVisible(False)
    # view.addItem(scatter_unity)
    
    print("[INFO] Stream running. Press 'c' to capture & send, 'q' to quit.")
    
    world_to_unity_cache = None
    analysis_results = []
    persistent_gt_corners = None
    persistent_reproj_corners = None

    while True:
        app.processEvents()
        
        success = grab_frames(cameras, image_mats, frames_batch, executor)
        if not success:
            continue
            
        if world_to_unity_cache is None:
            try:
                world_to_unity_cache = get_world_to_unity_transform(args.mode)
            except Exception:
                pass
                
        R_w2c, t_w2c = world_to_unity_cache if world_to_unity_cache else (None, None)
        received_corners = unity_corners.get_all()
        
        current_frame_analysis = {
            "cameras": {},
            "geometric_checks": {}
        }
        perform_analysis = False

        if received_corners and R_w2c is not None:
            # We want to analyze for each frame when we have Unity corners
            perform_analysis = True
            
            # Find OpenCV corners to compare
            # Map frames back to cpu for OpenCV
            import torch
            if isinstance(frames_batch, torch.Tensor):
                cv2_batch = frames_batch.cpu().numpy()
            else:
                cv2_batch = frames_batch
            corners_all = find_chessboard_corners(cv2_batch, config.CALIBRATION_BOARD_SIZE)
            extreme_corners_all = [get_extreme_corners(c, config.CALIBRATION_BOARD_SIZE) for c in corners_all]

            # 2) Geometric checks for rectangle on the corner received by Unity
            if len(received_corners) == 4:
                # Assuming index 0, 1, 2, 3 mapped correctly
                try:
                    u_tl = received_corners[0]["pos"]
                    u_tr = received_corners[1]["pos"]
                    u_bl = received_corners[2]["pos"]
                    u_br = received_corners[3]["pos"]
                    
                    v_top = u_tr - u_tl
                    v_bottom = u_br - u_bl
                    v_left = u_bl - u_tl
                    v_right = u_br - u_tr
                    
                    len_top = float(np.linalg.norm(v_top))
                    len_bottom = float(np.linalg.norm(v_bottom))
                    len_left = float(np.linalg.norm(v_left))
                    len_right = float(np.linalg.norm(v_right))
                    
                    dot_tl = float(np.dot(v_top, v_left))
                    dot_tr = float(np.dot(-v_top, v_right))
                    dot_bl = float(np.dot(v_bottom, -v_left))
                    dot_br = float(np.dot(-v_bottom, -v_right))

                    # Additional metrics for robustness
                    diag_1 = float(np.linalg.norm(u_br - u_tl))
                    diag_2 = float(np.linalg.norm(u_bl - u_tr))
                    
                    area = float(len_top * len_left) # rough area
                    
                    current_frame_analysis["geometric_checks"] = {
                        "len_top": len_top, "len_bottom": len_bottom,
                        "len_left": len_left, "len_right": len_right,
                        "dot_tl": dot_tl, "dot_tr": dot_tr,
                        "dot_bl": dot_bl, "dot_br": dot_br,
                        "diag_1": diag_1, "diag_2": diag_2,
                        "area": area,
                        "width_diff": abs(len_top - len_bottom),
                        "height_diff": abs(len_left - len_right)
                    }
                except KeyError:
                    pass
            
        for i, frame in enumerate(frames_batch):
            if hasattr(frame, 'cpu'):
                cv_frame = frame.cpu().numpy().copy()
            else:
                cv_frame = frame.copy()
            
            cam_analysis = {"distances_2d": {}}
            
            # Draw received Unity corners
            if received_corners and R_w2c is not None:
                P = proj_matrices[i]
                for idx, data in received_corners.items():
                    pos_unity = data["pos"]
                    hex_color = data["color"]
                    
                    # Convert hex "#RRGGBB" to BGR tuple
                    try:
                        h = hex_color.lstrip('#')
                        rgb = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
                        bgr = (rgb[2], rgb[1], rgb[0])
                    except Exception:
                        bgr = (255, 255, 255)
                        
                    # Unity local -> CV space
                    pos_cv = np.array([pos_unity[0], -pos_unity[1], pos_unity[2]], dtype=np.float32)
                    
                    # CV space -> World space
                    pos_world = R_w2c.T @ (pos_cv - t_w2c)
                    
                    pt_2d = project_point(P, pos_world)
                    if pt_2d:
                        cv2.circle(cv_frame, pt_2d, 6, bgr, -1)
                        cv2.circle(cv_frame, pt_2d, 8, (255, 255, 255), 1)
                        cv2.putText(cv_frame, f"U{idx}", (pt_2d[0]+10, pt_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
                        
                        # 1) Compute the distance between pt_2d and OpenCV detected corners
                        if perform_analysis and extreme_corners_all[i] is not None:
                            try:
                                cv_corner = extreme_corners_all[i][idx]
                                dist = float(np.linalg.norm(np.array(pt_2d) - cv_corner))
                                cam_analysis["distances_2d"][str(idx)] = dist
                                
                                # Draw line between them for visual debugging
                                cv2.line(cv_frame, pt_2d, (int(cv_corner[0]), int(cv_corner[1])), (0, 255, 255), 1)
                            except IndexError:
                                pass

            # --- Draw persistent ground truth and reprojected corners (after pressing 'c') ---
            if persistent_gt_corners is not None and persistent_reproj_corners is not None:
                if i < len(persistent_gt_corners) and i < len(persistent_reproj_corners):
                    gt_pts = persistent_gt_corners[i]
                    reproj_pts = persistent_reproj_corners[i]
                    if gt_pts is not None and reproj_pts is not None:
                        for corner_idx in range(min(len(gt_pts), len(reproj_pts))):
                            pt_gt = gt_pts[corner_idx]
                            pt_reproj = reproj_pts[corner_idx]
                            cv2.circle(cv_frame, (int(pt_gt[0]), int(pt_gt[1])), 8, (0, 255, 0), 2)
                            if pt_reproj is not None:
                                cv2.circle(cv_frame, (int(pt_reproj[0]), int(pt_reproj[1])), 8, (0, 0, 255), 2)
                                cv2.line(cv_frame, (int(pt_gt[0]), int(pt_gt[1])), (int(pt_reproj[0]), int(pt_reproj[1])), (0, 255, 255), 1)

            if perform_analysis:
                current_frame_analysis["cameras"][str(i)] = cam_analysis

            cv2.imshow(f"Camera {i}", cv2.resize(cv_frame, (640, 360)))
            
        if perform_analysis:
            analysis_results.append(current_frame_analysis)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            if hasattr(frames_batch, 'cpu'):
                cv2_frames = frames_batch.cpu().numpy()
            else:
                cv2_frames = frames_batch
                
            corners_all = find_chessboard_corners(cv2_frames, config.CALIBRATION_BOARD_SIZE)
            extreme_corners_all = [get_extreme_corners(c, config.CALIBRATION_BOARD_SIZE) for c in corners_all]
            
            # Display the 2D corners on videos briefly
            for i, (frame, corners, ext_corners) in enumerate(zip(cv2_frames, corners_all, extreme_corners_all)):
                display_frame = frame.copy()
                if corners is not None:
                    # Draw all found chessboard corners
                    cv2.drawChessboardCorners(display_frame, config.CALIBRATION_BOARD_SIZE, corners, True)
                if ext_corners is not None:
                    # Highlight the 4 extreme corners specifically
                    for pt in ext_corners:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 12, (0, 255, 255), -1)
                
                win_name = f"Camera {i}"
                cv2.imshow(win_name, cv2.resize(display_frame, (640, 360)))
            
            # Crucial: process window events so the drawing actually appears on screen
            cv2.waitKey(500) 
            
            corners_world = triangulate_corners(extreme_corners_all, proj_matrices)
            
            if corners_world is not None:                # Update 3D scene
                corners_pg = OpenCV2PyQtGraphAxisSystem(corners_world)
                scatter.setData(pos=corners_pg)
                scatter.setVisible(True)
                
                # Transform to Unity space and visualize
                corners_unity = transform_to_unity(corners_world, args.mode)
                
                # PyQtGraph is Z-up, Unity is Y-up. 
                # To visualize Unity coordinates gracefully in the same PyQtGraph view:
                # Unity(x, y, z) -> PyQtGraph(x, z, y) usually, 
                # but let's just apply the conversion to see them in the Unity-relative grid
                corners_unity_pg = OpenCV2PyQtGraphAxisSystem(corners_unity) 
                # scatter_unity.setData(pos=corners_unity_pg)
                # scatter_unity.setVisible(True)

                # Geometric checks for the rectangle
                tl, tr, bl, br = corners_world[0], corners_world[1], corners_world[2], corners_world[3]
                
                v_top = tr - tl
                v_bottom = br - bl
                v_left = bl - tl
                v_right = br - tr
                
                len_top = np.linalg.norm(v_top)
                len_bottom = np.linalg.norm(v_bottom)
                len_left = np.linalg.norm(v_left)
                len_right = np.linalg.norm(v_right)
                
                dot_tl = np.dot(v_top, v_left)
                dot_tr = np.dot(-v_top, v_right)   # tr -> tl vs tr -> br
                dot_bl = np.dot(v_bottom, -v_left) # bl -> br vs bl -> tl
                dot_br = np.dot(-v_bottom, -v_right) # br -> bl vs br -> tr

                print("\n[GEOMETRIC CHECKS]")
                print(f"  Widths  (Top, Bottom): {len_top:.4f}m, {len_bottom:.4f}m")
                print(f"  Heights (Left, Right): {len_left:.4f}m, {len_right:.4f}m")
                print(f"  Scalar products between adjacent edges: TL: {dot_tl:.6f}, TR: {dot_tr:.6f}, BL: {dot_bl:.6f}, BR: {dot_br:.6f}")

                print("\n[REPROJECTION ERROR (2D -> 3D -> 2D)]")
                persistent_gt_corners = extreme_corners_all
                persistent_reproj_corners = [[] for _ in proj_matrices]
                for i, P in enumerate(proj_matrices):
                    if extreme_corners_all[i] is not None:
                        errs = []
                        print(f"  Camera {i}:")
                        for corner_idx, pt_3d in enumerate(corners_world):
                            orig_2d = extreme_corners_all[i][corner_idx]
                            proj_2d = project_point(P, pt_3d)
                            persistent_reproj_corners[i].append(proj_2d)
                            if proj_2d:
                                dx = proj_2d[0] - orig_2d[0]
                                dy = proj_2d[1] - orig_2d[1]
                                dist = np.sqrt(dx**2 + dy**2)
                                errs.append(dist)
                                print(f"    Corner {corner_idx}: proj ({proj_2d[0]}, {proj_2d[1]}) vs orig ({orig_2d[0]:.1f}, {orig_2d[1]:.1f}) -> delta L2: {dist:.2f} px")
                        if errs:
                            print(f"    => Mean L2 Error Cam {i}: {np.mean(errs):.2f} px")
                    else:
                        persistent_reproj_corners[i] = None

                print(f"\n[SUCCESS] Corners in Unity tracking space:")
                color_names = ["green", "yellow", "red", "blue"] # Matches the visual order defined above
                for i, corner in enumerate(corners_unity):
                    color_name = color_names[i]
                    print(f"  Corner {i} ({color_name}): {corner}")
                    osc_client.send_message(f"/checkerboard/corner{i}", [float(corner[0]), float(corner[1]), float(corner[2]), color_name])
                print("[OSC] Messages sent.")
            else:
                print("[ERROR] Failed to triangulate corners")
                
    executor.shutdown(wait=False)
    if analysis_results:
        # Save results to json
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', args.mode))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'sender_corners_analysis.json')
        with open(out_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"[INFO] Saved analysis results to {out_path}")
        
    cv2.destroyAllWindows()
    close_cameras(cameras)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
