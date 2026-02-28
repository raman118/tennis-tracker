"""
Post-match analytics engine for the Tennis Player Tracking pipeline.

Computes rich per-player and match-level statistics from the raw tracking
data collected during video processing. This turns basic distance numbers
into a comprehensive performance breakdown that includes:

  - Speed percentiles (avg, median, max, P95)
  - Movement phase classification (sprint / jog / walk / stationary)
  - Court zone occupancy (time spent in each region of the court)
  - Court coverage percentage (what fraction of the court was visited)
  - Rally-level aggregated stats

Architecture decision: analytics are computed *after* the main loop finishes,
operating on the complete time-series data rather than per-frame incremental
updates. This is cleaner and allows backward-looking computations (e.g.,
rolling windows, percentiles) that would be awkward in a streaming model.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tennis_tracker import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Movement phase thresholds (meters per second)
# ---------------------------------------------------------------------------
STATIONARY_THRESHOLD_MPS: float = 0.3   # < 0.3 m/s ≈ standing still
WALK_THRESHOLD_MPS: float = 1.5         # 0.3 – 1.5 m/s ≈ walking / adjusting
JOG_THRESHOLD_MPS: float = 3.5          # 1.5 – 3.5 m/s ≈ jogging / shuffling
# > 3.5 m/s ≈ sprinting

# Court zone definitions (in meters, relative to ITF singles court)
ZONES = {
    "baseline_near":   (0.0, config.COURT_WIDTH_M, 18.0, config.COURT_LENGTH_M),
    "mid_court_near":  (0.0, config.COURT_WIDTH_M, 12.0, 18.0),
    "net_zone":        (0.0, config.COURT_WIDTH_M, 9.0, 12.0),
    "mid_court_far":   (0.0, config.COURT_WIDTH_M, 6.0, 9.0),
    "baseline_far":    (0.0, config.COURT_WIDTH_M, 0.0, 6.0),
}

# Left / Right split at court center
COURT_CENTER_X: float = config.COURT_WIDTH_M / 2.0

# Heatmap grid resolution
HEATMAP_BINS_X: int = 30
HEATMAP_BINS_Y: int = 60


@dataclass
class FrameSample:
    """One frame's worth of tracking data for analytics."""
    frame_idx: int
    meter_pos: Tuple[float, float]
    speed_mps: float  # instantaneous speed in m/s
    detected: bool


@dataclass
class MovementPhaseStats:
    """Time breakdown across movement phases."""
    stationary_pct: float = 0.0
    walking_pct: float = 0.0
    jogging_pct: float = 0.0
    sprinting_pct: float = 0.0
    stationary_seconds: float = 0.0
    walking_seconds: float = 0.0
    jogging_seconds: float = 0.0
    sprinting_seconds: float = 0.0


@dataclass
class ZoneStats:
    """Court zone occupancy breakdown."""
    baseline_near_pct: float = 0.0
    mid_court_near_pct: float = 0.0
    net_zone_pct: float = 0.0
    mid_court_far_pct: float = 0.0
    baseline_far_pct: float = 0.0
    deuce_side_pct: float = 0.0
    ad_side_pct: float = 0.0


@dataclass
class PlayerAnalytics:
    """Comprehensive analytics for a single player."""
    player_id: int
    total_distance_m: float = 0.0

    # Speed statistics (km/h for readability)
    avg_speed_kmh: float = 0.0
    median_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    p95_speed_kmh: float = 0.0

    # Movement phases
    movement: MovementPhaseStats = field(default_factory=MovementPhaseStats)

    # Zone occupancy
    zones: ZoneStats = field(default_factory=ZoneStats)

    # Court coverage
    court_coverage_pct: float = 0.0

    # Detection quality
    detection_rate_pct: float = 0.0
    total_frames_tracked: int = 0
    total_frames_lost: int = 0

    # Heatmap data (raw 2D histogram, not serialized to JSON directly)
    heatmap: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class MatchAnalytics:
    """Match-level analytics combining both players."""
    player_1: PlayerAnalytics = field(default_factory=lambda: PlayerAnalytics(player_id=1))
    player_2: PlayerAnalytics = field(default_factory=lambda: PlayerAnalytics(player_id=2))
    total_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0


class AnalyticsEngine:
    """Collects per-frame samples and computes post-match analytics.

    Usage:
        engine = AnalyticsEngine(fps=30.0)
        # During processing loop:
        engine.record(player_id=1, frame_idx=i, meter_pos=(mx, my),
                      speed_mps=v, detected=True)
        # After loop:
        analytics = engine.compute()
    """

    def __init__(self, fps: float, total_frames: int) -> None:
        self._fps = max(fps, 1.0)
        self._total_frames = total_frames
        self._samples: Dict[int, List[FrameSample]] = {1: [], 2: []}
        logger.info("AnalyticsEngine initialized (fps=%.1f, frames=%d).", fps, total_frames)

    def record(
        self,
        player_id: int,
        frame_idx: int,
        meter_pos: Optional[Tuple[float, float]],
        speed_mps: float,
        detected: bool,
    ) -> None:
        """Record one frame's data for a player."""
        if player_id not in (1, 2):
            return
        if meter_pos is None:
            meter_pos = (0.0, 0.0)
            detected = False

        self._samples[player_id].append(
            FrameSample(
                frame_idx=frame_idx,
                meter_pos=meter_pos,
                speed_mps=speed_mps,
                detected=detected,
            )
        )

    def compute(
        self,
        p1_distance: float,
        p2_distance: float,
    ) -> MatchAnalytics:
        """Compute comprehensive analytics from all recorded samples."""
        result = MatchAnalytics(
            total_frames=self._total_frames,
            fps=self._fps,
            duration_seconds=self._total_frames / self._fps,
        )

        result.player_1 = self._compute_player(1, p1_distance)
        result.player_2 = self._compute_player(2, p2_distance)

        logger.info(
            "Analytics computed — P1: avg %.1f km/h, coverage %.1f%% | "
            "P2: avg %.1f km/h, coverage %.1f%%",
            result.player_1.avg_speed_kmh,
            result.player_1.court_coverage_pct,
            result.player_2.avg_speed_kmh,
            result.player_2.court_coverage_pct,
        )

        return result

    def _compute_player(self, player_id: int, total_distance: float) -> PlayerAnalytics:
        """Compute analytics for a single player."""
        samples = self._samples[player_id]
        pa = PlayerAnalytics(player_id=player_id, total_distance_m=total_distance)

        if not samples:
            return pa

        # --- Detection quality ---
        detected_count = sum(1 for s in samples if s.detected)
        pa.total_frames_tracked = detected_count
        pa.total_frames_lost = len(samples) - detected_count
        pa.detection_rate_pct = round(100.0 * detected_count / max(len(samples), 1), 1)

        # --- Speed statistics ---
        # Only use frames where the player was actually detected
        detected_samples = [s for s in samples if s.detected]
        if detected_samples:
            speeds_mps = np.array([s.speed_mps for s in detected_samples])
            # Cap unrealistic speeds (>10 m/s ≈ 36 km/h is world-class sprint)
            speeds_mps = np.clip(speeds_mps, 0.0, 12.0)

            speeds_kmh = speeds_mps * 3.6
            pa.avg_speed_kmh = round(float(np.mean(speeds_kmh)), 1)
            pa.median_speed_kmh = round(float(np.median(speeds_kmh)), 1)
            pa.max_speed_kmh = round(float(np.max(speeds_kmh)), 1)
            pa.p95_speed_kmh = round(float(np.percentile(speeds_kmh, 95)), 1)

            # --- Movement phase breakdown ---
            pa.movement = self._compute_movement_phases(speeds_mps)

        # --- Zone occupancy ---
        pa.zones = self._compute_zone_occupancy(detected_samples)

        # --- Court coverage ---
        pa.court_coverage_pct, pa.heatmap = self._compute_court_coverage(detected_samples)

        return pa

    def _compute_movement_phases(self, speeds_mps: np.ndarray) -> MovementPhaseStats:
        """Classify each frame into movement phases and compute time breakdown."""
        n = len(speeds_mps)
        if n == 0:
            return MovementPhaseStats()

        stationary = np.sum(speeds_mps < STATIONARY_THRESHOLD_MPS)
        walking = np.sum(
            (speeds_mps >= STATIONARY_THRESHOLD_MPS) & (speeds_mps < WALK_THRESHOLD_MPS)
        )
        jogging = np.sum(
            (speeds_mps >= WALK_THRESHOLD_MPS) & (speeds_mps < JOG_THRESHOLD_MPS)
        )
        sprinting = np.sum(speeds_mps >= JOG_THRESHOLD_MPS)

        frame_duration = 1.0 / self._fps

        return MovementPhaseStats(
            stationary_pct=round(100.0 * stationary / n, 1),
            walking_pct=round(100.0 * walking / n, 1),
            jogging_pct=round(100.0 * jogging / n, 1),
            sprinting_pct=round(100.0 * sprinting / n, 1),
            stationary_seconds=round(float(stationary) * frame_duration, 1),
            walking_seconds=round(float(walking) * frame_duration, 1),
            jogging_seconds=round(float(jogging) * frame_duration, 1),
            sprinting_seconds=round(float(sprinting) * frame_duration, 1),
        )

    def _compute_zone_occupancy(self, samples: List[FrameSample]) -> ZoneStats:
        """Compute what percentage of time the player spent in each court zone."""
        if not samples:
            return ZoneStats()

        zone_counts: Dict[str, int] = defaultdict(int)
        deuce_count = 0
        ad_count = 0
        n = len(samples)

        for s in samples:
            mx, my = s.meter_pos
            # Depth zones
            for zone_name, (x_min, x_max, y_min, y_max) in ZONES.items():
                if x_min <= mx <= x_max and y_min <= my <= y_max:
                    zone_counts[zone_name] += 1
                    break

            # Lateral split
            if mx >= COURT_CENTER_X:
                deuce_count += 1
            else:
                ad_count += 1

        return ZoneStats(
            baseline_near_pct=round(100.0 * zone_counts.get("baseline_near", 0) / n, 1),
            mid_court_near_pct=round(100.0 * zone_counts.get("mid_court_near", 0) / n, 1),
            net_zone_pct=round(100.0 * zone_counts.get("net_zone", 0) / n, 1),
            mid_court_far_pct=round(100.0 * zone_counts.get("mid_court_far", 0) / n, 1),
            baseline_far_pct=round(100.0 * zone_counts.get("baseline_far", 0) / n, 1),
            deuce_side_pct=round(100.0 * deuce_count / n, 1),
            ad_side_pct=round(100.0 * ad_count / n, 1),
        )

    def _compute_court_coverage(
        self, samples: List[FrameSample]
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Compute court coverage percentage and generate a 2D heatmap.

        Court coverage is defined as the fraction of court bins that the player
        visited at least once. The heatmap is a 2D histogram of position density.
        """
        if not samples:
            return 0.0, None

        positions = np.array([s.meter_pos for s in samples])

        heatmap, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=[HEATMAP_BINS_X, HEATMAP_BINS_Y],
            range=[[0.0, config.COURT_WIDTH_M], [0.0, config.COURT_LENGTH_M]],
        )

        total_bins = HEATMAP_BINS_X * HEATMAP_BINS_Y
        visited_bins = np.sum(heatmap > 0)
        coverage_pct = round(100.0 * float(visited_bins) / total_bins, 1)

        return coverage_pct, heatmap

    @staticmethod
    def analytics_to_dict(analytics: MatchAnalytics) -> Dict:
        """Convert MatchAnalytics to a JSON-serializable dictionary.

        Excludes raw heatmap arrays (those are saved as images separately).
        """
        def player_dict(pa: PlayerAnalytics) -> Dict:
            return {
                "player_id": pa.player_id,
                "total_distance_m": round(pa.total_distance_m, 2),
                "speed": {
                    "avg_kmh": pa.avg_speed_kmh,
                    "median_kmh": pa.median_speed_kmh,
                    "max_kmh": pa.max_speed_kmh,
                    "p95_kmh": pa.p95_speed_kmh,
                },
                "movement_phases": {
                    "stationary": {
                        "pct": pa.movement.stationary_pct,
                        "seconds": pa.movement.stationary_seconds,
                    },
                    "walking": {
                        "pct": pa.movement.walking_pct,
                        "seconds": pa.movement.walking_seconds,
                    },
                    "jogging": {
                        "pct": pa.movement.jogging_pct,
                        "seconds": pa.movement.jogging_seconds,
                    },
                    "sprinting": {
                        "pct": pa.movement.sprinting_pct,
                        "seconds": pa.movement.sprinting_seconds,
                    },
                },
                "court_zones": {
                    "baseline_near_pct": pa.zones.baseline_near_pct,
                    "mid_court_near_pct": pa.zones.mid_court_near_pct,
                    "net_zone_pct": pa.zones.net_zone_pct,
                    "mid_court_far_pct": pa.zones.mid_court_far_pct,
                    "baseline_far_pct": pa.zones.baseline_far_pct,
                    "deuce_side_pct": pa.zones.deuce_side_pct,
                    "ad_side_pct": pa.zones.ad_side_pct,
                },
                "court_coverage_pct": pa.court_coverage_pct,
                "detection_quality": {
                    "detection_rate_pct": pa.detection_rate_pct,
                    "frames_tracked": pa.total_frames_tracked,
                    "frames_lost": pa.total_frames_lost,
                },
            }

        return {
            "analytics": {
                "player_1": player_dict(analytics.player_1),
                "player_2": player_dict(analytics.player_2),
            }
        }
