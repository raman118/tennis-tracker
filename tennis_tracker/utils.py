"""
Utility functions for the Tennis Player Tracking pipeline.

These are pure, stateless helper functions shared across multiple modules.
Each function is intentionally small, fully type-hinted, and has a docstring
that could serve as interview-ready documentation.

Architecture decision: keeping these as free functions (not methods) ensures
they remain testable in isolation and reusable without instantiating any class.
"""

import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Supported video file extensions for batch discovery. We include the three
# most common container formats used in broadcast and consumer video.
_VIDEO_EXTENSIONS: set = {".mp4", ".avi", ".mov"}


def ema_smooth(
    prev: Tuple[float, float],
    curr: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """Apply Exponential Moving Average (EMA) smoothing to a 2-D point.

    EMA is chosen over a simple moving average because it requires only the
    previous smoothed value (O(1) memory) and naturally weights recent
    observations higher, which is critical for real-time position tracking
    where latency matters.

    Formula per axis:  smoothed = alpha * current + (1 - alpha) * previous

    Args:
        prev: Previously smoothed (x, y) position.
        curr: Current raw (x, y) observation from the detector.
        alpha: Smoothing factor in (0, 1]. Higher = more responsive,
               lower = smoother but laggier.

    Returns:
        Smoothed (x, y) tuple.
    """
    # Apply EMA independently to each axis — they are uncorrelated in screen space
    sx = alpha * curr[0] + (1.0 - alpha) * prev[0]
    sy = alpha * curr[1] + (1.0 - alpha) * prev[1]
    return (sx, sy)


def euclidean_dist(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Compute Euclidean distance between two 2-D points.

    Uses math.hypot for numerical stability and speed (it avoids the
    intermediate squaring overflow risk of manual sqrt(dx² + dy²) for
    very large coordinate values).

    Args:
        p1: First point (x, y).
        p2: Second point (x, y).

    Returns:
        Scalar distance between p1 and p2.
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def pixel_to_meter(
    point: Tuple[float, float],
    H: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Transform a pixel-space point to real-world meter-space via homography.

    cv2.perspectiveTransform expects a (1, N, 2) float32 array and returns
    the same shape. We wrap this to provide a clean (x, y) → (mx, my) API.

    If H is None (calibration failed), we return the raw pixel coordinates
    as-is so the pipeline can still run in degraded "pixel_fallback" mode.

    Args:
        point: Pixel coordinates (px, py).
        H: 3×3 homography matrix, or None if calibration was skipped.

    Returns:
        (mx, my) in meters if H is available, else raw (px, py).
    """
    if H is None:
        # Degraded mode — distances will be in pixels, not meters
        return point

    # Reshape to the (1, 1, 2) array that cv2.perspectiveTransform requires
    pts = np.array([[[point[0], point[1]]]], dtype=np.float32)

    # Apply the 3×3 projective transformation (homogeneous division included)
    transformed = cv2.perspectiveTransform(pts, H)

    # Extract the (mx, my) result from the nested array structure
    mx, my = float(transformed[0][0][0]), float(transformed[0][0][1])
    return (mx, my)


def format_time(seconds: float) -> str:
    """Format a duration in seconds to M:SS display string.

    Broadcast-style time display without zero-padding the minutes,
    e.g. "2:07" instead of "00:02:07".

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        Formatted time string like "1:47".
    """
    # int() truncates toward zero, which is standard for elapsed-time display
    total_sec = int(seconds)
    minutes = total_sec // 60
    secs = total_sec % 60
    # Zero-pad seconds only (minutes are not padded in broadcast convention)
    return f"{minutes}:{secs:02d}"


def get_video_files(folder: str) -> List[Path]:
    """Discover all video files in a directory.

    Scans non-recursively for files with recognized video extensions.
    Results are sorted alphabetically for deterministic processing order
    across runs.

    Args:
        folder: Path to the directory to scan.

    Returns:
        Sorted list of Path objects pointing to video files.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Video directory not found: {folder}")

    # iterdir() is preferred over glob("*") because we want to match on
    # suffix explicitly, and iterdir avoids hidden-file edge cases on Linux
    videos = sorted(
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )

    logger.info("Found %d video file(s) in '%s'.", len(videos), folder)
    return videos


def resolve_video_path(stem: str, folder: str) -> Optional[Path]:
    """Find a video file by its stem name (without extension) in a folder.

    Useful when the user specifies a video by name without extension.

    Args:
        stem: File stem to search for (e.g. "tennis_video_assignment").
        folder: Directory to search in.

    Returns:
        Path to the matching video file, or None if not found.
    """
    for ext in _VIDEO_EXTENSIONS:
        candidate = Path(folder) / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None
