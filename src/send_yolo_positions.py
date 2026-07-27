"""
src.send_yolo_positions — Main YOLOv8 Pose Streaming Application
===================================================================

Main entry point for real-time streaming of triangulated 3D YOLOv8 pose keypoints from
multiple ZED cameras to Unity via OSC.

This script orchestrates the complete pipeline:
  1. Initialize multiple ZED cameras
  2. Run YOLOv8 pose estimation on batched frames
  3. Triangulate 2D keypoints to 3D world coordinates
  4. Transform to HMD coordinate space
  5. Stream 3D poses to Unity via OSC
  6. Optional: Record video with 2D poses and profile performance

Key Features:
  - Real-time multi-camera pose estimation and 3D triangulation
  - Concurrent frame grabbing for throughput optimization
  - Configurable calibration modes (barycenter, anchors, floor)
  - Optional video recording with FPS correction via ffmpeg
  - Comprehensive performance profiling (CPU, GPU, timing)
  - Display OpenCV windows for debugging
  - Graceful shutdown with resource cleanup

Command-Line Arguments:
  --mode {barycenter,anchors,floor}  Calibration mode (required)
  --display                          Show OpenCV camera windows with 2D poses
  --profile                          Enable performance profiling
  --record                           Record video with 2D pose overlays

Usage:
  # Basic streaming
  python -m src.send_yolo_positions --mode floor
  
  # With display and profiling
  python -m src.send_yolo_positions --mode floor --display --profile
  
  # Record video
  python -m src.send_yolo_positions --mode floor --record

Pipeline Steps:
  1. Load cameras, calibration data, and YOLO model
  2. Start OSC sender thread
  3. Main loop:
     a. Grab frames from all cameras concurrently
     b. Run YOLO pose estimation (batched, GPU-accelerated)
     c. Triangulate 2D keypoints to 3D world coordinates (DLT)
     d. Transform to HMD space
     e. Send to Unity via OSC
     f. Optional: Record and profile
  4. Graceful shutdown: stop sender, close cameras, finalize recordings

Performance Profiling Output:
  - Preprocess/Inference/Postprocess timing per frame
  - System RAM and GPU VRAM usage
  - Actual vs. nominal frame rates
  - Statistics: mean ± std across profiled frames

Video Recording:
  - Temporary raw MP4 files written with placeholder FPS
  - After run, actual FPS computed from wall-clock timestamps
  - Files re-encoded with ffmpeg to correct FPS for proper playback
  - Final files: record_cam_X_TIMESTAMP.mp4

Dependencies:
  - core.pose (YOLOv8 inference)
  - core.camera (ZED camera operations)
  - core.triangulation (3D reconstruction)
  - core.osc (OSC streaming)
  - core.transform (Calibration transforms)
  - config (Project configuration)
  - psutil, cv2, pyzed, numpy, torch (optional profiling)

Configuration (config.py):
  - CAMERA_SERIAL_NUMBERS: List of ZED camera serial numbers
  - CAMERA_FPS: Target camera frame rate (60 recommended)
  - YOLO_WEIGHTS: YOLOv8 model path
  - YOLO_FORMAT: Model format (pt, onnx, or engine)
  - HMD_OSC_IP, SEND_OSC_PORT: OSC destination
  - MIN_KEYPOINT_CONFIDENCE: Threshold for valid keypoints

Calibration Prerequisites:
  - Run calibration_to_world.py first (generates world-to-camera extrinsics)
  - Run hmd_calibration_to_world.py (generates world-to-HMD transforms)
  - Files saved in calibration_results/{mode}/ directories

Exit Conditions:
  - Press Ctrl+C in terminal
  - Press 'q' in OpenCV display window (if --display)
  - Any unhandled exception (logs error message)

Notes:
  - Warmup frames (100) before profiling to allow GPU stabilization
  - Frame rate determined by YOLO inference (typically 15-30 FPS for nano model)
  - OSC sender runs in separate thread at exactly 60 Hz regardless of inference rate
  - Video files saved in current working directory
"""

import sys
import os
import time
import cv2
import argparse
import numpy as np
import pyzed.sl as sl
from concurrent.futures import ThreadPoolExecutor

# Ensure src/ is in the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#core
from core.pose import load_model, estimate_poses
from core.camera import init_cameras, grab_frames, close_cameras
from core.triangulation import triangulate_multiview
from core.osc import OscSenderSleepBasedRateLimiter
from core.transform import get_world_to_hf_components, get_projection_matrices
#config
import config

# Try to import optional profiling modules
try:
    import psutil
    import torch
    HAS_PROFILING_LIBS = True
except ImportError:
    HAS_PROFILING_LIBS = False


def main():
    """Start the YOLO pose estimation loop and stream 3D points over OSC to Unity."""
    parser = argparse.ArgumentParser(description="Stream YOLO poses to Unity via OSC.")
    parser.add_argument("--mode", type=str, choices=["barycenter", "anchors", "floor"], required=True,
                        help="Calibration mode: barycenter, anchors, or floor")
    parser.add_argument("--display", action="store_true", help="Display OpenCV camera frames")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--record", action="store_true", help="Record video with 2D poses from both cameras at the actual achieved FPS")
    args = parser.parse_args()

    print("=== YOLO Pose to Unity OSC Streamer ===")
    print(f"Mode: {args.mode}, Display: {args.display}, Record: {args.record}")

    cam_serials = config.CAMERA_SERIAL_NUMBERS

    # Init Cameras
    cameras = init_cameras(cam_serials, resolution=sl.RESOLUTION.HD720, fps=config.CAMERA_FPS)
    if len(cameras) < 2:
        print("[ERROR] Need at least 2 cameras for triangulation.")
        close_cameras(cameras)
        return

    # Projection matrices (world → camera image plane)
    try:
        proj_matrices = get_projection_matrices(cameras)
        print(f"[CALIB] Loaded {len(proj_matrices)} projection matrices.")
    except Exception as e:
        print(f"[ERROR] Loading projection matrices: {e}")
        close_cameras(cameras)
        return

    # World → Hf transform (loaded once from precomputed JSON)
    R_W_to_Hf, t_W_to_Hf = get_world_to_hf_components(args.mode)

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
    yolo_model = load_model(config.YOLO_WEIGHTS, config.YOLO_FORMAT)

    # Init OSC Sender
    sender = OscSenderSleepBasedRateLimiter(ip=config.HMD_OSC_IP, port=config.SEND_OSC_PORT, target_hz=60.0)
    sender.start()

    # ── Init Video Recorders ──────────────────────────────────────────
    # NOTE on FPS: we no longer assume 60 fps. cv2.VideoWriter requires an
    # fps value up front, but we don't actually know the real achieved fps
    # until the loop has been running for a while (it's bottlenecked by
    # YOLO inference, not by camera capture). So we:
    #   1. Write frames to temporary "raw" files using a placeholder fps.
    #   2. Track a real wall-clock timestamp for every frame we write.
    #   3. After the loop ends, compute the *actual* fps from
    #      (frames written) / (elapsed wall-clock time) per camera.
    #   4. Re-encode each raw file with ffmpeg using that real fps, so the
    #      final mp4 plays back at the correct speed.
    video_writers = []
    raw_video_paths = []
    final_video_paths = []
    record_frame_timestamps = []  # wall-clock time of each written frame
    record_start_time = None

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        PLACEHOLDER_FPS = 30.0  # arbitrary; corrected during finalize step
        for i in range(len(cameras)):
            raw_filename = f"record_cam_{i}_{timestamp}_raw.mp4"
            final_filename = f"record_cam_{i}_{timestamp}.mp4"
            writer = cv2.VideoWriter(raw_filename, fourcc, PLACEHOLDER_FPS, (w, h))
            video_writers.append(writer)
            raw_video_paths.append(raw_filename)
            final_video_paths.append(final_filename)
        print(f"[RECORD] Video recording enabled. Files will be finalized as record_cam_X_{timestamp}.mp4 "
              "with FPS corrected to match the actual capture rate.")

    print("\n[STREAM] Running... Press Ctrl+C in terminal (or 'q' in CV window if --display) to exit.")

    # ── Send-rate measurement ─────────────────────────────────────────
    _rate_counter   = 0
    _rate_window    = 2.0
    _rate_last_time = time.perf_counter()

    # ── Profiling accumulators ───────────────────────────────────────────────
    prof_pre_times   = []   # preprocess phase
    prof_inf_times   = []   # inference phase
    prof_post_times  = []   # postprocess / keypoint extraction phase
    prof_ram_usages  = []
    prof_vram_allocs = []
    prof_vram_res    = []
    frame_count  = 0
    warmup_frames  = 100

    try:
        while True:
            success = grab_frames(cameras, mats, frames_batch, executor)
            if not success:
                continue

            frame_count += 1

            result = estimate_poses(yolo_model, frames_batch, args.profile)
            if args.profile:
                poses_2d, timings = result
            else:
                poses_2d, timings = result, {}

            if args.profile:
                if frame_count == warmup_frames + 1:
                    print('[PROFILING] Warmup done. Starting profiling...')
                elif frame_count < warmup_frames:
                    if frame_count % 20 == 0:
                        print(f"[PROFILING] Warmup {frame_count}/{warmup_frames}...")
                else:
                    prof_pre_times.append(timings["preprocess_ms"])
                    prof_inf_times.append(timings["inference_ms"])
                    prof_post_times.append(timings["postprocess_ms"])
                    if HAS_PROFILING_LIBS:
                        process = psutil.Process(os.getpid())
                        prof_ram_usages.append(process.memory_info().rss / (1024 ** 2))
                        if torch.cuda.is_available():
                            prof_vram_allocs.append(torch.cuda.memory_allocated() / (1024 ** 2))
                            prof_vram_res.append(torch.cuda.memory_reserved() / (1024 ** 2))

            if len(proj_matrices) == len(cameras):
                pose_3d_world = triangulate_multiview(poses_2d, proj_matrices)

                valid_mask = np.any(pose_3d_world != 0, axis=1)
                kpts_3d_hf = np.zeros_like(pose_3d_world)
                kpts_3d_hf[valid_mask] = pose_3d_world[valid_mask] @ R_W_to_Hf.T + t_W_to_Hf

                sender.update(kpts_3d_hf)

            # ── Record Frames with 2D Poses drawn on them ──────────────────
            if args.record:
                if record_start_time is None:
                    record_start_time = time.perf_counter()
                record_frame_timestamps.append(time.perf_counter())

                for i, writer in enumerate(video_writers):
                    frame_source = frames_batch[i]
                    if hasattr(frame_source, 'cpu'):
                        frame = frame_source.cpu().numpy().copy()
                    else:
                        frame = frame_source.copy()

                    # Draw the 2D keypoints onto the copied frame for recording
                    kpts = poses_2d[i]
                    for x, y, _ in kpts:
                        if x != 0 and y != 0:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                    # Ensure contiguous array layout for VideoWriter
                    if not frame.flags['C_CONTIGUOUS']:
                        frame = np.ascontiguousarray(frame)

                    writer.write(frame)

            # Rate measurement
            _rate_counter += 1
            t_now = time.perf_counter()
            if t_now - _rate_last_time >= _rate_window:
                fps = _rate_counter / (t_now - _rate_last_time)
                _rate_counter = 0
                _rate_last_time = t_now

            # Optional debug visualization
            if args.display:
                for i in range(len(cameras)):
                    frame_source = frames_batch[i]
                    if hasattr(frame_source, 'cpu'):
                        frame = frame_source.cpu().numpy().copy()
                    else:
                        frame = frame_source.copy()

                    kpts = poses_2d[i]
                    for x, y, _ in kpts:
                        if x != 0 and y != 0:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    cv2.imshow(f"Cam {i}", cv2.resize(frame, (640, 360)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        # Stop and release resources
        executor.shutdown(wait=False)
        sender.stop()
        close_cameras(cameras)

        # Release video writers so files don't corrupt, then correct FPS
        if args.record:
            for writer in video_writers:
                writer.release()

            n_frames = len(record_frame_timestamps)
            if n_frames >= 2 and record_start_time is not None:
                elapsed = record_frame_timestamps[-1] - record_frame_timestamps[0]
                actual_fps = (n_frames - 1) / elapsed if elapsed > 0 else 30.0
                print(f"[RECORD] Captured {n_frames} frames over {elapsed:.2f}s "
                      f"→ actual FPS = {actual_fps:.2f}")

                print("[RECORD] Re-encoding with corrected FPS via ffmpeg...")
                for raw_path, final_path in zip(raw_video_paths, final_video_paths):
                    ok = finalize_recording(raw_path, final_path, actual_fps)
                    if ok:
                        print(f"[RECORD] Saved '{final_path}' at {actual_fps:.2f} FPS.")
            else:
                print("[RECORD][WARN] Not enough frames captured to determine actual FPS; "
                      "raw files left as-is with placeholder FPS.")

        if args.display:
            cv2.destroyAllWindows()

        if args.profile:
            if len(prof_inf_times) > 0:
                pre_mean  = float(np.mean(prof_pre_times))
                pre_std   = float(np.std(prof_pre_times))
                inf_mean  = float(np.mean(prof_inf_times))
                inf_std   = float(np.std(prof_inf_times))
                post_mean = float(np.mean(prof_post_times))
                post_std  = float(np.std(prof_post_times))
                total_mean = pre_mean + inf_mean + post_mean

                print("\n" + "="*50)
                print("              PROFILING REPORT")
                print("="*50)
                print(f"Frames analyzed  : {len(prof_inf_times)}")
                print(f"Warmup frames    : {warmup_frames}")
                print("-"*50)
                print(f"  Preprocess     : {pre_mean:7.2f} ± {pre_std:.2f} ms")
                print(f"  Inference      : {inf_mean:7.2f} ± {inf_std:.2f} ms")
                print(f"  Postprocess    : {post_mean:7.2f} ± {post_std:.2f} ms")
                print(f"  {'─'*38}")
                print(f"  Total          : {total_mean:7.2f} ms")

                ram_mean = ram_std = None
                vram_alloc_mean = vram_alloc_std = None
                vram_res_mean   = vram_res_std   = None

                if HAS_PROFILING_LIBS and len(prof_ram_usages) > 0:
                    ram_mean = float(np.mean(prof_ram_usages))
                    ram_std  = float(np.std(prof_ram_usages))
                    print(f"\nSystem RAM (RSS) : {ram_mean:.1f} ± {ram_std:.1f} MB")

                    if len(prof_vram_allocs) > 0:
                        vram_alloc_mean = float(np.mean(prof_vram_allocs))
                        vram_alloc_std  = float(np.std(prof_vram_allocs))
                        vram_res_mean   = float(np.mean(prof_vram_res))
                        vram_res_std    = float(np.std(prof_vram_res))
                        print(f"GPU VRAM Alloc   : {vram_alloc_mean:.1f} ± {vram_alloc_std:.1f} MB")
                        print(f"GPU VRAM Reserved: {vram_res_mean:.1f} ± {vram_res_std:.1f} MB")
                else:
                    print("\nSkipped RAM/VRAM profiling. Check 'psutil' / 'torch'.")

                print("="*50 + "\n")
            else:
                print(f"[PROFILE] No frames collected yet (still in warmup at frame {frame_count}/{warmup_frames}).")

if __name__ == "__main__":
    main()