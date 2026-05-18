"""
render_scene_new.py
Gracefully renders the 3D scene with the calibrated cameras, world origin, and properly mapped HMD poses.
"""

import sys
import os
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import glob

# Add the root structure to system path to import modules normally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config 

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

def draw_axes(view, pos, R_world, label_text=None, scale=0.3, width=3):
    """
    Draw 3D axes at a given position and orientation (in OpenCV world space).
    pos: 3D position in world (numpy array)
    R_world: Rotation matrix of the object in the world frame (numpy array)
    scale: length of the axis lines
    """
    x_axis = pos + R_world @ np.array([scale, 0, 0])
    y_axis = pos + R_world @ np.array([0, scale, 0])
    z_axis = pos + R_world @ np.array([0, 0, scale])
    
    axes_pts = np.array([
        pos, x_axis,
        pos, y_axis,
        pos, z_axis
    ], dtype=np.float32)
    
    # Convert OpenCV coordinates to PyQtGraph coordinates for visualization
    axes_pts = OpenCV2PyQtGraphAxisSystem(axes_pts)
    
    axes_colors = np.array([
        [1, 0, 0, 1], [1, 0, 0, 1],  # X: Red
        [0, 1, 0, 1], [0, 1, 0, 1],  # Y: Green
        [0, 0, 1, 1], [0, 0, 1, 1]   # Z: Blue
    ], dtype=np.float32)
    
    axis_line = gl.GLLinePlotItem(pos=axes_pts, color=axes_colors, mode='lines', width=width)
    view.addItem(axis_line)

    if label_text:
        label_pos = OpenCV2PyQtGraphAxisSystem(pos)
        text_item = gl.GLTextItem(pos=label_pos, text=label_text, color=(255, 255, 255, 200))
        view.addItem(text_item)

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
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

def get_cv_to_world_transform(mode):
    # Load extrinsics from JSON
    extrinsics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results', mode, 'hmd_poses_in_world.json'))
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    xr_camera = ext.get("xr_camera")
    if xr_camera is None:
        raise ValueError("Extrinsics for xr_camera not found in JSON.")
        
    R_xr_world = np.array(xr_camera['R'])
    t_xr_world = np.array(xr_camera['t']).reshape(3)
    
    # Load snapshot to get Unity pose at calibration
    snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
    json_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.json"))
    if not json_files:
        raise ValueError("No Snapshot json found.")
        
    latest_snapshot = sorted(json_files)[-1]
    with open(latest_snapshot, 'r') as f:
        snap = json.load(f)
        
    # Get calibration xrCamera pose directly from snapshot
    pos = snap["xrCamera_position"]
    rot = snap["xrCamera_rotation"]
    pos_xr_unity = np.array([pos["x"], pos["y"], pos["z"]])
    rot_xr_unity_q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
    
    # 1. Standardize Coordinates
    pos_xr_cv, R_xr_cv = unity_to_cv(pos_xr_unity, rot_xr_unity_q)
    
    # 2. Compute the static spatial offset
    R_c2w = R_xr_world @ R_xr_cv.T
    t_c2w = t_xr_world - R_c2w @ pos_xr_cv
    
    return R_c2w, t_c2w

def map_hmd_pose_to_world(pos_unity, quat_unity, R_c2w, t_c2w):
    """ Apply the offset to map unity pose into OpenCV world """
    pos_cv, R_cv = unity_to_cv(pos_unity, quat_unity)
    
    pos_world = R_c2w @ pos_cv + t_c2w
    R_world = R_c2w @ R_cv
    return pos_world, R_world

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Static Camera Scene Calibration (World Mapped)")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors"], required=True, help="Calibration mode: barycenter or anchors")
    args = parser.parse_args()

    app = pg.mkQApp("3D Scene Visualization")
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle(f"Static Camera Scene Calibration (Mode: {args.mode})")
    view.setCameraPosition(distance=5, elevation=20, azimuth=45)
    
    # Add a reference grid
    grid = gl.GLGridItem()
    grid.scale(1, 1, 1)
    view.addItem(grid)
    
    # 1. Draw World Origin
    draw_axes(view, np.array([0, 0, 0], dtype=np.float32), np.eye(3), label_text="World Origin", scale=1.0, width=5)
    print(f"[INFO] Rendered World Origin (Mode: {args.mode})")
    
    calibration_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration_results'))
    mode_dir = os.path.join(calibration_dir, args.mode)
    
    # 2. Draw ZED Cameras
    cameras_ext_path = os.path.join(calibration_dir, 'world_to_camera_extrinsics.json')
    if os.path.exists(cameras_ext_path):
        with open(cameras_ext_path, 'r') as f:
            cam_ext = json.load(f)
        
        for k, v in cam_ext.items():
            if k.startswith("camera_"):
                R_mat = np.array(v['R'])
                t = np.array(v['t']).reshape(3)
                R_world = R_mat.T
                pos_world = -R_world @ t
                cam_label = k.replace("camera_", "ZED ")
                draw_axes(view, pos_world, R_world, label_text=cam_label, scale=0.4, width=3)
                print(f"[INFO] Rendered {cam_label}")

    # 3. Calculate Transform and Draw Mapped HMD Snapshot Poses
    try:
        R_c2w, t_c2w = get_cv_to_world_transform(args.mode)
        
        snapshot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshot'))
        json_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.json"))
        if json_files:
            latest_snapshot = sorted(json_files)[-1]
            with open(latest_snapshot, 'r') as f:
                snap_data = json.load(f)

            def get_un_pos(data): return np.array([data['x'], data['y'], data['z']])
            def get_un_rot(data): return np.array([data['x'], data['y'], data['z'], data['w']])

            if "position" in snap_data and "rotation" in snap_data:
                pos = get_un_pos(snap_data["position"])
                rot = get_un_rot(snap_data["rotation"])
                pos_w, rot_w = map_hmd_pose_to_world(pos, rot, R_c2w, t_c2w)
                draw_axes(view, pos_w, rot_w, label_text="Mapped Head Center", scale=0.5, width=4)
                print("[INFO] Rendered Mapped Head Center")

            if "left_camera_position" in snap_data and "left_camera_rotation" in snap_data:
                pos = get_un_pos(snap_data["left_camera_position"])
                rot = get_un_rot(snap_data["left_camera_rotation"])
                pos_w, rot_w = map_hmd_pose_to_world(pos, rot, R_c2w, t_c2w)
                draw_axes(view, pos_w, rot_w, label_text="Mapped Left Eye", scale=0.4, width=4)
                print("[INFO] Rendered Mapped Left Eye")
                
            if "xrCamera_position" in snap_data and "xrCamera_rotation" in snap_data:
                pos = get_un_pos(snap_data["xrCamera_position"])
                rot = get_un_rot(snap_data["xrCamera_rotation"])
                pos_w, rot_w = map_hmd_pose_to_world(pos, rot, R_c2w, t_c2w)
                draw_axes(view, pos_w, rot_w, label_text="Mapped xrCamera", scale=0.3, width=4)
                print("[INFO] Rendered Mapped xrCamera from Snapshot")

    except Exception as e:
        print(f"[ERROR] Could not map HMD snapshot poses: {e}")

    # 4. Draw Calibrated HMD from Extrinsics file (Should perfectly overlap mapped snapshot poses)
    hmd_ext_path = os.path.join(mode_dir, 'world_to_hmd_extrinsics.json')
    if os.path.exists(hmd_ext_path):
        with open(hmd_ext_path, 'r') as f:
            hmd_ext = json.load(f)
            
        for k, v in hmd_ext.items():
            if k == "left_eye" or k == "xr_camera":
                continue
            R_mat = np.array(v['R'])
            t = np.array(v['t']).reshape(3)
            R_world = R_mat.T
            pos_world = -R_world @ t
            draw_axes(view, pos_world, R_world, label_text=f"Calibrated {k}", scale=0.6, width=2)
            
            print(f"[INFO] Rendered Calibrated {k}")

    print("\n[INFO] Scene rendered successfully. Close the window to exit.")

    try:
        pg.exec()
    except KeyboardInterrupt:
        print("[DEBUG] Shutting down...")

if __name__ == "__main__":
    main()
