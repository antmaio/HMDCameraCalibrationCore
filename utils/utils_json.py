"""
utils.utils_json — JSON I/O Utilities
========================================

Provides simple JSON serialization and file management helpers.

This module contains utility functions for:
  - Loading JSON data from files
  - Saving JSON data with pretty indentation
  - Ensuring directory paths exist before writing

Key Features:
  - Clean, readable JSON output with 4-space indentation
  - Automatic directory creation before saving
  - Simple exception handling for file I/O
  - Supports any JSON-serializable Python objects

Functions:
  load_json(path) -> data: Load JSON from file
  save_json(path, data) -> None: Save JSON to file (creates directory if needed)
  ensure_dir(path) -> None: Create directory if it doesn't exist
  _ensure_dir(path) -> None: Internal private version

Usage:
  # Load calibration data
  calib_data = load_json('calibration_results/extrinsics.json')
  
  # Save results with auto directory creation
  save_json('results/output.json', {'x': 1, 'y': 2})
  # Creates 'results/' directory if it doesn't exist
  
  # Ensure directory exists
  ensure_dir('output_data')

Dependencies:
  - json (standard library)
  - os (standard library)

Notes:
  - JSON output uses indent=4 for readability
  - ensure_dir uses os.makedirs with exist_ok=True (safe for existing dirs)
  - _ensure_dir is private; use through save_json or ensure_dir
  - All paths are UTF-8 encoded

Common Use Cases:
  - Saving calibration results: save_json('calibration_results/extrinsics.json', transforms)
  - Loading camera parameters: K = load_json('camera_intrinsics.json')
  - Preparing output directories: ensure_dir('output/results')
"""

import json
import os

def _ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)

def ensure_dir(path: str) -> None:
    """Public wrapper: Create a directory if it does not already exist."""
    _ensure_dir(path)

def save_json(path: str, data: object) -> None:
    """Save JSON data to a file with pretty indentation."""
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> object:
    """Load JSON data from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

