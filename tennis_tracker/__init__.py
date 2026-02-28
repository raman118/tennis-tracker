"""
Tennis Player Tracking & Distance Measurement package.

This package provides a modular pipeline for detecting, tracking, and
measuring the distance traveled by tennis players in broadcast video.

Modules:
    config      -- All tunable constants and device configuration.
    utils       -- Pure helper functions (EMA, distance, transforms).
    calibration -- Court corner detection and homography computation.
    detector    -- YOLOv8x + ByteTrack person detection and filtering.
    tracker     -- Player identity locking and distance accumulation.
    visualizer  -- Video overlay rendering (trails, boxes, HUD).
    mini_court  -- Bird's-eye view court diagram overlay.
    analytics   -- Post-match analytics (speed, zones, coverage).
    heatmap     -- Court heatmap image generation.
"""

from tennis_tracker.calibration import CourtCalibrator
from tennis_tracker.detector import PlayerDetector, Detection
from tennis_tracker.tracker import PlayerTracker, PlayerState
from tennis_tracker.visualizer import Visualizer
from tennis_tracker.analytics import AnalyticsEngine, MatchAnalytics

__all__ = [
    "CourtCalibrator",
    "PlayerDetector",
    "Detection",
    "PlayerTracker",
    "PlayerState",
    "Visualizer",
    "AnalyticsEngine",
    "MatchAnalytics",
]
