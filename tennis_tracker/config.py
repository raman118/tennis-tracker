"""
Configuration module for the Tennis Player Tracking pipeline.

All tunable constants are defined here with inline comments explaining
both the value chosen and the reasoning behind it. This eliminates
magic numbers from all other modules and provides a single source of
truth for the entire pipeline's behavior.

Architecture decision: A flat module of constants (rather than a dataclass
or YAML file) was chosen because every consumer needs compile-time access
to these values, and the pipeline has no need for runtime config swapping.
"""

import logging
import torch

# ---------------------------------------------------------------------------
# Logging — configured once at import time so every module inherits it
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Device selection — prefer CUDA, fall back to CPU with a logged warning
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE: str = "cuda"
    logging.getLogger(__name__).info("CUDA device detected — using GPU acceleration.")
else:
    DEVICE: str = "cpu"
    logging.getLogger(__name__).warning(
        "CUDA not available — falling back to CPU. Inference will be significantly slower."
    )

# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------
# YOLOv8x is the extra-large variant with the highest mAP on COCO.
# For a sports broadcast with small, distant players we need every bit of
# accuracy the model can offer; the speed penalty is acceptable for offline
# video processing.
YOLO_MODEL: str = "yolov8x.pt"

# COCO class ID 0 corresponds to 'person'. We filter out all 79 other COCO
# classes because only human detections are relevant for player tracking.
PERSON_CLASS_ID: int = 0

# Detections below this confidence are too noisy to trust. 0.5 is the
# standard industry threshold that balances recall (catching real players)
# against precision (rejecting phantom detections from motion blur, ads, etc).
CONF_THRESHOLD: float = 0.5

# ByteTrack is built into the ultralytics library and does not require a
# separate re-identification model download (unlike DeepSORT). It uses a
# Kalman filter + IoU-based association which is both fast and robust on
# fixed-camera sports footage where player motion is smooth frame-to-frame.
TRACKER_CONFIG: str = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Player identity locking
# ---------------------------------------------------------------------------
# During the first WARMUP_FRAMES frames we observe which track IDs appear
# most frequently. At 30 fps this is ~1 second — enough for ByteTrack to
# stabilize its Kalman priors before we commit to locking two IDs.
WARMUP_FRAMES: int = 30

# A tennis singles match has exactly 2 on-court players.
MAX_PLAYERS: int = 2

# ---------------------------------------------------------------------------
# Position smoothing
# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) alpha controls the trade-off between
# responsiveness and noise suppression. alpha=0.3 means 30% weight on the
# new observation and 70% on the running average. This suppresses the ±2-4
# pixel bounding-box jitter caused by video compression artifacts while
# still tracking real player movement with minimal temporal lag.
EMA_ALPHA: float = 0.3

# ---------------------------------------------------------------------------
# Distance accumulation
# ---------------------------------------------------------------------------
# After EMA smoothing, residual jitter is ~0.5-1 pixel. When transformed
# through the homography, this becomes ~0.02-0.04 m of phantom displacement
# per frame. The threshold of 0.05 m ensures we only accumulate movement
# that exceeds this noise floor. A real tennis player's minimum observable
# step at broadcast resolution is well above this threshold.
MIN_DIST_THRESHOLD_M: float = 0.05

# ---------------------------------------------------------------------------
# Court geometry (ITF standard singles court)
# ---------------------------------------------------------------------------
# These are the official International Tennis Federation dimensions for a
# singles court measured baseline-to-baseline and sideline-to-sideline.
# Used as the real-world target rectangle for the homography transform.
COURT_WIDTH_M: float = 8.23    # sideline to sideline (meters)
COURT_LENGTH_M: float = 23.77  # baseline to baseline (meters)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Number of historical foot-point positions retained for the trajectory
# trail overlay. At 30 fps, 60 frames = 2 seconds of visible trail.
TRAIL_LENGTH: int = 60

# Percentage by which to expand the court polygon for is_inside_court checks.
# The auto-detected corners lie on the court lines, but players at the near
# baseline often have their foot point a few pixels BELOW the line due to
# perspective and bounding box imprecision. A 15% expansion ensures players
# at baseline edges aren't falsely classified as outside the court.
COURT_MARGIN_PERCENT: float = 0.15

# Maximum pixel distance for nearest-centroid re-ID recovery.
# When ByteTrack assigns a new track ID after an occlusion, we try to recover
# by assigning the new detection to the closest known player. But if the
# detection is too far away (> 200px), it's likely a different person entirely
# (ball boy, umpire) who slipped through the court filter. In that case we
# refuse the assignment to prevent identity contamination.
MAX_RECOVERY_DIST_PX: float = 200.0

# BGR color tuples — OpenCV uses BGR channel order, not RGB.
# Player 1: steel blue (distinctive on the blue hard-court surface)
P1_COLOR: tuple = (219, 112, 50)   # BGR
# Player 2: warm red (high contrast against blue court and P1's blue)
P2_COLOR: tuple = (50, 80, 219)    # BGR

# Opacity of the semi-transparent HUD background rectangle.
# 0.65 makes the text clearly readable while still showing the court
# underneath, avoiding the "opaque block" look of a fully solid panel.
HUD_ALPHA: float = 0.65

# ---------------------------------------------------------------------------
# Paths — relative to project root; resolved at runtime via pathlib
# ---------------------------------------------------------------------------
VIDEOS_DIR: str = "videos"
OUTPUTS_DIR: str = "outputs"

# ---------------------------------------------------------------------------
# Detection scoring weights (must sum to 1.0)
# Used by the multi-signal player-vs-non-player scoring system to weight
# four independent evidence signals when deciding which detections are
# real on-court players.
# ---------------------------------------------------------------------------
W_COURT_ZONE: float   = 0.40   # court zone position is the strongest signal
W_ASPECT: float       = 0.25   # bounding box aspect ratio (tall narrow = player)
W_SIZE: float         = 0.20   # bounding box size relative to frame height
W_CONFIDENCE: float   = 0.15   # raw YOLO confidence (already 0.0–1.0)

# ---------------------------------------------------------------------------
# Court zone boundaries (as fraction of frame dimensions)
# These define the horizontal and vertical zones used to score how likely
# a detection's position corresponds to an on-court player versus a
# sideline official, ball boy, or spectator.
# ---------------------------------------------------------------------------
COURT_H_TOP: float    = 0.25   # top 25% of frame height = far baseline zone
COURT_H_BOTTOM: float = 0.75   # below 75% of frame height = near baseline zone
COURT_V_LEFT: float   = 0.15   # left 15% of frame width = sideline/crowd zone
COURT_V_RIGHT: float  = 0.85   # right 85% of frame width = sideline/crowd zone

# ---------------------------------------------------------------------------
# Velocity gate for re-ID recovery
# ---------------------------------------------------------------------------
# Maximum realistic player displacement in pixels between consecutive frames.
# At 30fps, a sprinting player (~10 m/s) moves ~40px/frame at broadcast
# resolution. 80px provides a generous safety margin without allowing
# assignments to detections on the opposite side of the court.
MAX_PIXELS_PER_FRAME: int = 80

# ---------------------------------------------------------------------------
# Distance sanity cap
# ---------------------------------------------------------------------------
# Maximum realistic distance a player can travel in one frame at 30fps.
# A full sprint covers ~0.33 m/frame; 1.5 m is impossible and indicates a
# detection jump (ID swap, re-ID failure) rather than real movement.
MAX_DELTA_PER_FRAME_M: float = 1.5
