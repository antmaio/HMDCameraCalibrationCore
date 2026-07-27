"""
core.osc — OSC Pose Streaming
================================

Handles background thread-based Open Sound Control (OSC) broadcasting of 3D pose data.

This module provides a timer-based OSC sender for streaming 3D keypoint data to external
receivers (e.g., Unity) at a fixed target frequency while maintaining thread-safe updates.

Key Features:
  - Background daemon thread that broadcasts pose data at a fixed frequency (default 60 Hz)
  - Thread-safe pose updates via lock-protected slot
  - Strict timing using sleep-based rate limiting to maintain target frequency
  - Repeats last known pose if no new data is provided (useful for receivers expecting constant stream)
  - Flattens (17, 3) pose arrays into 51-element OSC messages

Classes:
  OscSenderSleepBasedRateLimiter: Main sender with continuous broadcast behavior
  
Usage:
  sender = OscSenderSleepBasedRateLimiter(ip='127.0.0.1', port=5005, target_hz=60)
  sender.start()
  # In main loop:
  sender.update(pose_3d)  # (17, 3) numpy array
  # Cleanup:
  sender.stop()

Dependencies:
  - pythonosc (for UDP client)
  - numpy
  - threading

Notes:
  - OSC address: /yolo/pose3d
  - Pose format: [x0, y0, z0, x1, y1, z1, ..., x16, y16, z16]
  - Thread-safe for concurrent updates from main inference loop
"""

import threading
import time
import numpy as np
from pythonosc.udp_client import SimpleUDPClient


class OscSenderSleepBasedRateLimiter:
    """
    Timer-based background thread that broadcasts 3D pose data over OSC.

    BEHAVIOR:
    - Always broadcasts at the target frequency (e.g., 60 Hz), even if no new data is received.
    - If no new `update()` is called, it repeatedly sends the last known pose.
    - Useful for receivers that expect a constant stream of data to maintain state or smoothing.
    """

    OSC_ADDRESS = "/yolo/pose3d"

    def __init__(self, ip: str = "127.0.0.1", port: int = 5005, target_hz: float = 60.0):
        self._ip = ip
        self._port = port
        self._frame_time = 1.0 / target_hz
        self._target_hz = target_hz

        self._lock = threading.Lock()
        self._latest_pose = np.zeros((17, 3), dtype=np.float32)
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sender thread."""
        self._running = True
        self._thread.start()
        print(f"[OSC] Sender started → {self._ip}:{self._port} @ {self._target_hz} Hz")

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._running = False
        self._thread.join()
        print("[OSC] Sender stopped.")

    def update(self, pose_3d: np.ndarray) -> None:
        """
        Thread-safe update of the pose data to be sent.

        Args:
            pose_3d: (17, 3) numpy array of 3D keypoints.
        """
        with self._lock:
            self._latest_pose = pose_3d.copy()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        client = SimpleUDPClient(self._ip, self._port)

        while self._running:
            t_start = time.perf_counter()

            with self._lock:
                pose = self._latest_pose.copy()

            # Flatten (17, 3) → 51 floats: [x0,y0,z0, x1,y1,z1, ..., x16,y16,z16]
            osc_data = pose.flatten().tolist()
            client.send_message(self.OSC_ADDRESS, osc_data)

            # Strict timing — sleep only the remaining frame budget
            elapsed = time.perf_counter() - t_start
            sleep_time = self._frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

'''
class OscSenderLatestOnlySlot:
    """
    Slot-based background thread that broadcasts 3D pose data over OSC.

    BEHAVIOR:
    - Only broadcasts when new data is available in the slot (consumed on send).
    - If `update()` is called faster than the target frequency, older data is overwritten and discarded.
    - If `update()` is called slower than the target frequency, the thread sleeps and sends nothing until data arrives.
    - Useful for reducing network traffic and ensuring the receiver only processes fresh updates without redundancy.
    """

    def __init__(self, ip: str, port: int, target_hz: float = 60.0):
        self._addr = (ip, port)
        self._interval = 1.0 / target_hz
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._latest = None          # single slot — always overwritten, never queued
        self._running = False
        self._thread = None

    def update(self, kpts_3d: np.ndarray):
        """Called from the main loop. Overwrites the slot; old data is discarded."""
        with self._lock:
            self._latest = kpts_3d.flatten().astype(np.float32)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._sock.close()

    def _send_loop(self):
        next_send = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            sleep_for = next_send - now
            if sleep_for > 0:
                time.sleep(sleep_for)

            with self._lock:
                payload = self._latest
                self._latest = None          # consume the slot

            if payload is not None:
                self._sock.sendto(self._build_osc(payload), self._addr)

            # Advance deadline absolutely — prevents drift accumulation
            next_send += self._interval
            # If we've fallen more than one interval behind, reset
            if time.perf_counter() > next_send:
                next_send = time.perf_counter() + self._interval

    @staticmethod
    def _build_osc(floats: np.ndarray) -> bytes:
        addr = b"/pose\x00\x00\x00"
        n = len(floats)
        tags = ("," + "f" * n).encode() + b"\x00"
        tags += b"\x00" * ((-len(tags)) % 4)
        data = struct.pack(f">{n}f", *floats)
        return addr + tags + data
'''