"""Utility helpers shared across calibration, pose, and streaming scripts."""

import glob
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------------
# DIRECTORY & FILE MANAGEMENT
# ----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)

# ----------------------------------------------------------------------------
# JSON 
# ----------------------------------------------------------------------------
def save_json(path: str, data: object) -> None:
    """Save JSON data to a file with pretty indentation."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> object:
    """Load JSON data from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------------
# CALIBRATION
# ----------------------------------------------------------------------------
def build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix from focal and principal point parameters."""
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def build_intrinsics_from_zed(left_cam) -> np.ndarray:
    """Build a camera intrinsic matrix from ZED left-camera calibration parameters."""
    return build_intrinsics_matrix(left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy)

def generate_3d_checkerboard_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    """Generate 3D points for a chessboard pattern lying on the Z=0 plane."""
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def get_snapshot_files(snapshot_dir: str) -> tuple[str, str]:
    """Find the latest snapshot image and its paired JSON metadata file."""
    png_files = glob.glob(os.path.join(snapshot_dir, "Snapshot_*.png"))
    if not png_files:
        raise FileNotFoundError(f"No Snapshot_*.png found in {snapshot_dir}")
    img_path = sorted(png_files)[-1]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(snapshot_dir, f"{base_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON metadata for {img_path}")
    return img_path, json_path
    
def load_latest_snapshot_json(snapshot_dir: str) -> dict:
    """Load the latest snapshot metadata JSON from a snapshot directory."""
    _, json_path = get_snapshot_files(snapshot_dir)
    return load_json(json_path)

# ----------------------------------------------------------------------------
# TRANSFORMATION
# ----------------------------------------------------------------------------
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def unity_to_cv(pos: np.ndarray, rot_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Unity world pose to OpenCV coordinate space pose."""
    pos = np.asarray(pos, dtype=np.float64)
    rot_q = np.asarray(rot_q, dtype=np.float64)
    pos_cv = np.array([pos[0], -pos[1], pos[2]], dtype=np.float64)
    R_u = quaternion_to_rotation_matrix(rot_q)
    M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    R_cv = M @ R_u @ M
    return pos_cv, R_cv


def cv_to_unity(pos_cv: np.ndarray) -> np.ndarray:
    """Convert a position from OpenCV coordinate space back to Unity coordinate space."""
    pos_cv = np.asarray(pos_cv, dtype=np.float64)
    if pos_cv.ndim == 1:
        return np.array([pos_cv[0], -pos_cv[1], pos_cv[2]], dtype=np.float32)
    out = pos_cv.copy().astype(np.float32)
    out[:, 1] = -out[:, 1]
    return out


def compute_camera_pose_in_world(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the camera pose in the world frame from world-to-camera extrinsics."""
    R_world = R.T
    t_world = -R_world @ t.reshape(3, 1)
    return R_world, t_world




