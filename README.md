# HMD Camera Calibration

This repository contains scripts and tools for performing Head-Mounted Display (HMD) camera calibration, incorporating YOLOv8 for pose estimation and OSC for data transmission.

## Project Structure

- `src/`: Contains the main source code for calibration to world coordinates, camera handling, triangulation, and OSC transmission (`send_yolo_positions.py`, `osc_sender.py`).
- `debug/`: Contains internal scripts for evaluating calibration, 2D/3D rendering, and general debugging.
- `config.py`: Main configuration variables and parameters for the calibration and application logic.

## Functionality
- **Pose Estimation:** Utilizes YOLOv8 (pose variants) to estimate tracking points.
- **World & HMD Calibration:** Scripts available to calibrate external cameras to world coordinates and to register HMD points tracking to world references.
- **OSC Integration:** Support implemented to broadcast coordinates and YOLO poses via the Open Sound Control (OSC) protocol.

## Data Directories (Ignored in Git)
Note that generated outputs such as calibration results, video/image frames, and system snapshots are intentionally ignored out of source control to keep the repository unbloated:
- `calibration_results/`
- `frames/`
- `screenshot_for_report/`
- `snapshot/`
