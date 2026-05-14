"""
config.py
Central configuration for the pose capture pipeline.
Edit this file to match your hardware setup.
"""

# ------------------------------------------------------------------
# ZED Camera Settings
# ------------------------------------------------------------------

# Replace with your actual ZED serial numbers
CAMERA_SERIAL_NUMBERS: list[int] = [
    35429635,
    38579233 #TODO: Replace with real serial numbers after testing each camera individually
    #33333333, #TODO: Replace with real serial numbers after testing each camera individually
]

CAMERA_FPS: int = 60

#For calibration 
CALIBRATION_BOARD_SIZE = (5, 6)  # Number of inner corners per chessboard row and columnchs
SQUARE_SIZE:float = 0.25  # Size of a square in meters

# ------------------------------------------------------------------
# OSC / Unity Settings
# ------------------------------------------------------------------

HMD_OSC_IP: str = "10.104.202.43"
SEND_OSC_PORT: int = 5005
LISTEN_OSC_PORT: int = 5010
OSC_TARGET_HZ: float = 60.0

# ------------------------------------------------------------------
# YOLO Settings
# ------------------------------------------------------------------

YOLO_WEIGHTS: str   = "yolov8n-pose"
USE_ONNX: bool      = False  # Set to True if you have an ONNX export of the model for faster inference on some platforms
#COCO keypoint indices:
# 0: Nose
# 1: Left Eye
# 2: Right Eye
# 3: Left Ear
# 4: Right Ear
# 5: Left Shoulder
# 6: Right Shoulder
# 7: Left Elbow
# 8: Right Elbow
# 9: Left Wrist
# 10: Right Wrist
# 11: Left Hip
# 12: Right Hip
# 13: Left Knee
# 14: Right Knee
# 15: Left Ankle
# 16: Right Ankle

# Coco keypoint order for reference
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), 
    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), 
    (3, 5), (4, 6)
]

# ------------------------------------------------------------------
# Triangulation Settings
# ------------------------------------------------------------------

# Set USE_MOCK_TRIANGULATION = False once you have real calibration data.
USE_MOCK_TRIANGULATION: bool = True

# Extrinsic calibration data per camera (world frame).
# Fill these in after running your ChArUco / stereo calibration.
# Each entry: {"R": [[...3x3...]], "t": [tx, ty, tz]}
EXTRINSICS: list[dict] = [
    {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0.0, 0.0, 0.0]},   # Camera 0 (world origin)
    {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.5, 0.0, 0.0]},   # Camera 1 — REPLACE
    {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [-1.5, 0.0, 0.0]},  # Camera 2 — REPLACE
]
