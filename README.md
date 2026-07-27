# HMD Camera Calibration

This repository contains scripts and tools for performing Head-Mounted Display (HMD) camera calibration, incorporating YOLOv8 for pose estimation and OSC for data transmission.

## Functionality
- **World & HMD Calibration:** Scripts available to calibrate external cameras to world coordinates and to register HMD points tracking to world references.
- **3D Pose Estimation:** Utilizes YOLOv8 (pose variants) and multiview DLT to estimate 3D positions of body joints 
- **OSC Integration:** Broadcast YOLO poses via the Open Sound Control (OSC) protocol to Unity app.

## Usage
Follow this workflow to set up calibration and begin streaming poses to your Unity application:

1. **Camera Calibration to World Frame**
   Builds extrinsics of all cameras in the scene relative to the world reference frame.
   - `--display`: Display intermediate calibration images (useful for debug)
   - `--out_dir`: Directory to save calibration results (default: `calibration_results`)
   ```
   python -m src.calibration_to_world [--display] [--out_dir calibration_results]
   ```


2. **HMD Calibration to World Frame**
   Builds extrinsics of the HMD left camera in the same world reference frame.
   - `--mode` (required): Calibration mode: `barycenter`, `anchors`, `floor`, or `both`
   - `--display`: Display calibration images
   - `--snapshot_dir`: Directory with Snapshot files (default: `snapshot`)
   - `--out_dir`: Directory to save calibration results (default: `calibration_results`)
   ```
   python -m src.hmd_calibration_to_world --mode {barycenter,anchors,floor,both} [--display] [--snapshot_dir snapshot] [--out_dir calibration_results]
   ```

3. **Compute World ↔ HMD Transform**
   Computes the transformation between the world reference frame and the HMD's native reference frame (Hf, where HMD left camera is initially expressed).
   ```
   python -m src.transform
   ```

4. **Stream YOLO Poses to Unity**
   Tracking mode refers to the method to fix Hf tracking space in Unity. E.g., `floor` is related to the floor detection and means that Hf axis system is attached to the ground.   
   ```
   python -m src.send_yolo_positions --mode {barycenter,anchors,floor} [--display] [--profile] [--record]
   ```
   Sends 3D poses estimated from camera videos to your Unity application via OSC.
   - `--mode` (required): Tracking mode: `barycenter`, `anchors`, or `floor`
   - `--display`: Display OpenCV camera frames
   - `--profile`: Enable performance profiling
   - `--record`: Record video with 2D poses from both cameras at the actual achieved FPS