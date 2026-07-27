"""
utils.utils_calibration — Calibration Utilities
==================================================

Provides helper functions for camera calibration workflows.

This module contains utility functions for:
  - Building camera intrinsic matrices from focal length and principal point
  - Extracting intrinsics from ZED camera objects
  - Generating 3D checkerboard point patterns for calibration
  - Finding and loading snapshot images and their JSON metadata

Key Features:
  - Constructs standard 3×3 intrinsic matrices in OpenCV format
  - Works with ZED SDK camera calibration parameters
  - Generates collinear 3D points for checkerboard calibration targets
  - Locates latest snapshot files by timestamp
  - Loads associated JSON metadata for snapshot intrinsics

Functions:
  build_intrinsics_matrix(fx, fy, cx, cy) -> K: Build 3×3 intrinsic matrix
  build_intrinsics_from_zed(left_cam) -> K: Extract from ZED camera object
  generate_3d_checkerboard_points(board_size, square_size) -> points: Generate calibration points
  get_snapshot_files(snapshot_dir) -> (img_path, json_path): Find latest snapshot pair
  load_latest_snapshot_json(snapshot_dir) -> metadata: Load snapshot metadata

Usage:
  # Build intrinsic matrix
  K = build_intrinsics_matrix(fx=525.0, fy=525.0, cx=320.0, cy=240.0)
  
  # Extract from ZED camera
  cam_info = camera.get_camera_information()
  left_cam = cam_info.camera_configuration.calibration_parameters.left_cam
  K = build_intrinsics_from_zed(left_cam)
  
  # Generate checkerboard points for 9x6 board with 30mm squares
  objp = generate_3d_checkerboard_points((9, 6), 30.0)  # Returns (54, 3)
  
  # Find and load snapshot
  img_path, json_path = get_snapshot_files('snapshot')
  metadata = load_latest_snapshot_json('snapshot')

Dependencies:
  - numpy
  - glob, os
  - utils.utils_json (load_json)

Intrinsic Matrix Format:
  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
  where fx, fy = focal lengths (pixels), cx, cy = principal point (pixels)

Checkerboard Format:
  Generates N points on Z=0 plane with spacing=square_size
  For board_size=(h, w), returns h*w 3D points: (x, y, 0)
  
Snapshot Files:
  Expected format: snapshot_dir/Snapshot_*.png and Snapshot_*.json
  Returns latest file by alphabetical sort (assumes timestamp in filename)
"""

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

