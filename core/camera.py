"""
core.camera — ZED Camera Management
======================================

Handles multi-camera ZED initialization, frame capture, and video recording finalization.

This module provides functions for:
  - Initializing multiple ZED stereo cameras by serial number with configurable resolution and FPS
  - Concurrent frame grabbing from all cameras using thread pools for parallel acquisition
  - Graceful camera shutdown and resource cleanup
  - Video recording finalization with FPS correction via ffmpeg re-encoding

Key Features:
  - Thread-safe concurrent frame capture from multiple ZED cameras
  - Support for both CPU (numpy) and GPU (CUDA torch) frame buffers
  - Automatic intrinsic parameter logging (focal length, principal point)
  - FPS correction for recorded video files to match actual capture rate

Usage:
  cameras = init_cameras([12345, 67890], fps=60)  # Initialize 2 cameras
  frames = grab_frames(cameras, mats, batch, executor)  # Capture frames
  close_cameras(cameras)  # Cleanup

Dependencies:
  - pyzed.sl (ZED SDK)
  - cv2 (OpenCV)
  - torch (PyTorch, optional for GPU buffers)
  - ffmpeg (for video finalization)
"""

import pyzed.sl as sl
import cv2
import torch
import os
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess

def init_cameras(serial_numbers: list[int], resolution=sl.RESOLUTION.HD720, fps: int = 60) -> list[sl.Camera]:
    """
    Initialize and open ZED cameras by serial number.

    Args:
        serial_numbers: List of ZED camera serial numbers.
        resolution: ZED resolution enum (default HD720).
        fps: Target camera frame rate (default 60).

    Returns:
        List of opened sl.Camera instances.
    """
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth for performance

    cameras = []
    print("[ZED] Initializing cameras...")

    for sn in serial_numbers:
        init_params.set_from_serial_number(sn)
        cam = sl.Camera()

        if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print(f"\033[93m[ZED] WARNING: Failed to open camera {sn}\033[0m")
        else:
            cam_info = cam.get_camera_information()
            intrinsics = cam_info.camera_configuration.calibration_parameters.left_cam
            print(
                f"[ZED] Camera {sn} opened — "
                f"fx:{intrinsics.fx:.2f}, fy:{intrinsics.fy:.2f}, "
                f"cx:{intrinsics.cx:.2f}, cy:{intrinsics.cy:.2f}"
            )

        cameras.append(cam)
        
    return cameras


def grab_frames(cameras: list[sl.Camera], image_mats: list[sl.Mat], out_batch, executor: ThreadPoolExecutor) -> bool:
    """
    Grab one BGR frame from each camera concurrently.

    Args:
        cameras: List of opened sl.Camera instances.
        image_mats: Pre-allocated list of sl.Mat buffers (one per camera).
        out_batch: Pre-allocated 4D numpy array or Torch Tensor (num_cameras, height, width, 3) for BGR data.
        executor: ThreadPoolExecutor to run grabs concurrently.

    Returns:
        True if all cameras successfully grabbed a frame, False otherwise.
    """
    futures = []
    for i, cam in enumerate(cameras):
        futures.append(executor.submit(_grab_single, cam, image_mats[i], out_batch, i))

    success = True
    for f in futures:
        if not f.result():
            success = False
            
    return success


def close_cameras(cameras: list[sl.Camera]) -> None:
    """
    Close all ZED cameras gracefully.

    Args:
        cameras: List of sl.Camera instances to close.
    """
    for cam in cameras:
        cam.close()
    print("[ZED] All cameras closed.")

def finalize_recording(raw_path: str, final_path: str, actual_fps: float) -> bool:
    """
    Re-encode a raw recording (written with a placeholder fps) into a file
    whose container fps matches the *actual* measured capture rate, so
    playback speed matches real elapsed time.

    Returns True on success, False if ffmpeg is unavailable or the call fails
    (in which case the raw file is kept as-is and a warning is printed).
    """
    if shutil.which("ffmpeg") is None:
        print("[RECORD][WARN] ffmpeg not found on PATH — cannot correct FPS. "
              f"Keeping raw file '{raw_path}' (was written assuming an incorrect FPS).")
        return False

    # Clamp to something sane in case of measurement noise (e.g. very first frames).
    safe_fps = max(1.0, min(actual_fps, 240.0))

    cmd = [
        "ffmpeg", "-y",
        "-r", f"{safe_fps:.4f}",   # interpret input frames at the measured rate
        "-i", raw_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        final_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(raw_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[RECORD][WARN] ffmpeg re-encode failed for '{raw_path}': {e}. Keeping raw file.")
        return False

def _grab_single(cam: sl.Camera, mat: sl.Mat, out_array, index: int) -> bool:
    """Helper to grab a single frame and convert it in a worker thread."""
    if cam.grab() == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        bgr = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
        if isinstance(out_array, torch.Tensor):
            # Direct copy to pre-allocated CUDA tensor memory slot
            out_array[index] = torch.from_numpy(bgr)
        else:
            out_array[index] = bgr
        return True
    return False

