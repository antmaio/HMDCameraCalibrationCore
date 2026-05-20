"""
pose.py
Handles YOLOv8 pose estimation on batched camera frames.
"""

import config
import numpy as np
from ultralytics import YOLO
import torch

import torch.nn.functional as F
import math
import config

NUM_KEYPOINTS = 17


def load_model(weights: str = "yolov8n-pose", yolo_format: str | bool = "pt") -> YOLO:
    """
    Load a YOLOv8 pose model.

    Args:
        weights: Path to model weights file without extension.
        yolo_format: Model format, one of 'pt', 'onnx', or 'engine'.
                     For backwards compatibility, a boolean is accepted
                     where True is treated as 'onnx'.

    Returns:
        Loaded YOLO model instance.
    """
    if isinstance(yolo_format, bool):
        yolo_format = "onnx" if yolo_format else "pt"
    yolo_format = str(yolo_format).lower()
    if yolo_format not in ("pt", "onnx", "engine"):
        raise ValueError(
            "yolo_format must be one of 'pt', 'onnx', or 'engine'"
        )

    weights = f"{weights}.{yolo_format}"
    print(f"[YOLO] Loading model from '{weights}'...")
    model = YOLO(weights)
    if yolo_format == "pt":
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("[YOLO] Model ready.")
    return model


def _letterbox_batch(batch: torch.Tensor, target: int = 640, stride: int = 32, pad_value: float = 114/255.0) -> torch.Tensor:
    """
    Replicates YOLO's internal letterbox preprocess on a BHWC uint8 tensor.
    Returns a BCHW float32 tensor ready for model inference.

    Args:
        batch:      (B, H, W, 3) uint8 tensor in BGR order, on any device.
        target:     Longest side target (default 640).
        stride:     Model stride — output dims are rounded to a multiple of this (default 32).
        pad_value:  Letterbox fill value, normalized (default 114/255).

    Returns:
        (B, 3, new_H, new_W) float32 tensor, where new_H and new_W are
        the smallest multiples of `stride` that fit the letterboxed image.
    """
    B, H, W, C = batch.shape

    # ── 1. Compute scale and new unpadded size ──────────────────────
    scale     = min(target / H, target / W)
    new_h     = round(H * scale)
    new_w     = round(W * scale)

    # ── 2. Padded canvas sizes ───────────
    pad_h = target
    pad_w = target

    # ── 3. BHWC uint8 → BCHW float32, normalize ─────────────────────
    #    permute is zero-copy on contiguous tensors
    x = batch.permute(0, 3, 1, 2).contiguous().float().div(255.0)  # (B, C, H, W)

    # ── 4. Resize (bilinear, align_corners=False matches cv2.INTER_LINEAR) ──
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # ── 5. Pad to stride-aligned canvas (right + bottom only, like YOLO) ────
    pad_right  = pad_w - new_w
    pad_bottom = pad_h - new_h
    x = F.pad(x, (0, pad_right, 0, pad_bottom), value=pad_value)  # (B, C, pad_h, pad_w)

    return x


def estimate_poses(model: YOLO, frames_batch: torch.Tensor|list[np.ndarray]|np.ndarray) -> list[np.ndarray]:
    """
    Run batched pose estimation.

    Args:
        model:        Loaded YOLO pose model.
        frames_batch: (num_cameras, H, W, 3) uint8 torch.Tensor, or list of np.ndarray.

    Returns:
        List of (17, 2) numpy arrays of 2D keypoints per camera.
        Zero arrays for frames with no detected person.
    """
    # ----------------------------------------------------------------
    # ── ONNX or NumPy fallback (List of arrays) ─────────────────────
    # ----------------------------------------------------------------

    if config.USE_ONNX:
        if isinstance(frames_batch, list) or isinstance(frames_batch, np.ndarray):
            frames = list(frames_batch) if isinstance(frames_batch, np.ndarray) else frames_batch
            with torch.inference_mode():
                results = model(frames, verbose=False, stream=True)
                
            poses = []
            num_cameras = len(frames)
            for result in results:
                if result.keypoints is not None and len(result.keypoints.xy) > 0:
                    kpts = result.keypoints.xy[0].cpu().numpy()
                    poses.append(kpts.astype(np.float32))
                else:
                    poses.append(np.zeros((17, 2), dtype=np.float32))
                    
            while len(poses) < num_cameras:
                poses.append(np.zeros((17, 2), dtype=np.float32))
                
            return poses

    # ----------------------------------------------------------------
    # ── PyTorch Tensor Fast Path ─────────────────────────────────────
    # ----------------------------------------------------------------

    else:
        if not isinstance(frames_batch, torch.Tensor):
            raise ValueError(f"frames_batch must be a torch.Tensor, got {type(frames_batch)}")
        if frames_batch.ndim != 4 or frames_batch.shape[-1] != 3:
            raise ValueError(f"Expected shape (B, H, W, 3), got {frames_batch.shape}")

        B = frames_batch.shape[0]

        # ── Replicate YOLO's internal preprocess ────────────────────────
        preprocessed = _letterbox_batch(frames_batch, target=config.IMAGE_TARGET_SIZE)  # (B, 3, pH, pW) float32 if 
        # ── Forward pass — bypass YOLO's own preprocess ─────────────────
        # Passing an already-normalized BCHW float tensor skips the internal
        # letterbox so YOLO won't double-process the input.
        with torch.inference_mode():
            results = model(preprocessed, verbose=False, stream=True)

        # ── Extract keypoints ────────────────────────────────────────────
        num_cameras = B
        poses = []
        for result in results:  # one result per image in the batch
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                # Take the highest-confidence detection (index 0)
                kpts = result.keypoints.xy[0].cpu().numpy()  # (17, 2) in letterboxed space

                # ── Invert letterbox scale to get original pixel coords ──
                scale = min(config.IMAGE_TARGET_SIZE / frames_batch.shape[1], config.IMAGE_TARGET_SIZE / frames_batch.shape[2])
                kpts = kpts / scale

                poses.append(kpts.astype(np.float32))
            else:
                poses.append(np.zeros((17, 2), dtype=np.float32))

        # Pad output if stream yielded fewer results than B
        while len(poses) < num_cameras:
            poses.append(np.zeros((17, 2), dtype=np.float32))

        return poses