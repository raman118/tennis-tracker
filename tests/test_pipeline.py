"""
Unit tests for the Tennis Player Tracking pipeline.

Tests the core algorithmic components in isolation: EMA smoothing,
distance computation, homography transforms, detection scoring,
movement phase classification, and analytics computation.

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis_tracker import config
from tennis_tracker.utils import ema_smooth, euclidean_dist, pixel_to_meter, format_time
from tennis_tracker.detector import Detection, PlayerDetector
from tennis_tracker.analytics import (
    AnalyticsEngine,
    FrameSample,
    STATIONARY_THRESHOLD_MPS,
    WALK_THRESHOLD_MPS,
    JOG_THRESHOLD_MPS,
)


# ===========================================================================
# utils.py tests
# ===========================================================================

class TestEmaSmooth:
    """Tests for the Exponential Moving Average smoothing function."""

    def test_alpha_zero_returns_previous(self):
        """Alpha=0 means 100% previous value, 0% current."""
        result = ema_smooth((10.0, 20.0), (50.0, 80.0), alpha=0.0)
        assert result == (10.0, 20.0)

    def test_alpha_one_returns_current(self):
        """Alpha=1 means 0% previous value, 100% current."""
        result = ema_smooth((10.0, 20.0), (50.0, 80.0), alpha=1.0)
        assert result == (50.0, 80.0)

    def test_alpha_half_averages(self):
        """Alpha=0.5 means equal blend of previous and current."""
        result = ema_smooth((0.0, 0.0), (10.0, 20.0), alpha=0.5)
        assert result == pytest.approx((5.0, 10.0))

    def test_typical_alpha(self):
        """Alpha=0.3 (the default EMA_ALPHA in config)."""
        prev = (100.0, 200.0)
        curr = (110.0, 210.0)
        result = ema_smooth(prev, curr, alpha=0.3)
        expected_x = 0.3 * 110.0 + 0.7 * 100.0
        expected_y = 0.3 * 210.0 + 0.7 * 200.0
        assert result == pytest.approx((expected_x, expected_y))

    def test_convergence_over_many_steps(self):
        """After many EMA steps toward the same target, result converges."""
        pos = (0.0, 0.0)
        target = (100.0, 100.0)
        for _ in range(200):
            pos = ema_smooth(pos, target, alpha=0.3)
        assert pos == pytest.approx(target, abs=0.01)


class TestEuclideanDist:
    """Tests for Euclidean distance computation."""

    def test_zero_distance(self):
        assert euclidean_dist((5.0, 5.0), (5.0, 5.0)) == 0.0

    def test_unit_distance(self):
        assert euclidean_dist((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_pythagorean_triple(self):
        assert euclidean_dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_negative_coordinates(self):
        assert euclidean_dist((-3.0, -4.0), (0.0, 0.0)) == pytest.approx(5.0)

    def test_symmetry(self):
        d1 = euclidean_dist((1.0, 2.0), (4.0, 6.0))
        d2 = euclidean_dist((4.0, 6.0), (1.0, 2.0))
        assert d1 == pytest.approx(d2)


class TestPixelToMeter:
    """Tests for homography-based pixel-to-meter transformation."""

    def test_none_homography_returns_raw_pixels(self):
        """When H is None (calibration failed), raw pixels are returned."""
        result = pixel_to_meter((500.0, 300.0), None)
        assert result == (500.0, 300.0)

    def test_identity_homography(self):
        """Identity matrix should return the input point unchanged."""
        H = np.eye(3, dtype=np.float64)
        result = pixel_to_meter((5.0, 10.0), H)
        assert result == pytest.approx((5.0, 10.0), abs=0.01)

    def test_known_homography_corners(self):
        """Verify that corner points map correctly through a known homography."""
        # Define pixel corners and meter corners for a simple mapping
        pixel_pts = np.array([
            [0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]
        ], dtype=np.float32)
        meter_pts = np.array([
            [0.0, 0.0], [8.23, 0.0], [8.23, 23.77], [0.0, 23.77]
        ], dtype=np.float32)

        import cv2
        H, _ = cv2.findHomography(pixel_pts, meter_pts)

        # Top-left corner should map to (0, 0)
        result = pixel_to_meter((0.0, 0.0), H)
        assert result == pytest.approx((0.0, 0.0), abs=0.1)

        # Bottom-right corner should map to (8.23, 23.77)
        result = pixel_to_meter((100.0, 200.0), H)
        assert result == pytest.approx((8.23, 23.77), abs=0.1)


class TestFormatTime:
    """Tests for time formatting."""

    def test_zero(self):
        assert format_time(0) == "0:00"

    def test_under_minute(self):
        assert format_time(45) == "0:45"

    def test_exact_minute(self):
        assert format_time(60) == "1:00"

    def test_mixed(self):
        assert format_time(125) == "2:05"

    def test_large_value(self):
        assert format_time(3661) == "61:01"


# ===========================================================================
# detector.py tests
# ===========================================================================

class TestDetectionScoring:
    """Tests for the multi-signal detection scoring system."""

    @staticmethod
    def _make_detection(
        foot_point=(500.0, 400.0),
        bbox=(450, 200, 550, 400),
        confidence=0.9,
        track_id=1,
    ) -> Detection:
        return Detection(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            foot_point=foot_point,
        )

    def test_center_court_player_scores_high(self):
        """A centered player with good shape should score > 0.7."""
        det = self._make_detection(
            foot_point=(500.0, 400.0),
            bbox=(470, 200, 530, 400),  # h/w=200/60≈3.3, height_frac=200/720≈0.28
            confidence=0.9,
        )
        score = PlayerDetector.compute_detection_score(det, (720, 1080, 3))
        assert score > 0.7

    def test_edge_detection_scores_low(self):
        """A detection at the far edge of the frame should score lower."""
        det = self._make_detection(
            foot_point=(50.0, 50.0),  # Far top-left corner
            bbox=(30, 20, 70, 80),    # Small square-ish box
            confidence=0.6,
        )
        score = PlayerDetector.compute_detection_score(det, (720, 1080, 3))
        assert score < 0.5

    def test_score_range(self):
        """Score should always be between 0 and 1."""
        for conf in [0.1, 0.5, 0.9]:
            for foot in [(100, 100), (500, 400), (900, 600)]:
                det = self._make_detection(foot_point=foot, confidence=conf)
                score = PlayerDetector.compute_detection_score(det, (720, 1080, 3))
                assert 0.0 <= score <= 1.0

    def test_weights_sum_to_one(self):
        """Config scoring weights must sum to 1.0 for proper normalization."""
        total = config.W_COURT_ZONE + config.W_ASPECT + config.W_SIZE + config.W_CONFIDENCE
        assert total == pytest.approx(1.0)


class TestFilterToPlayers:
    """Tests for the player filtering logic."""

    def test_empty_detections(self):
        result = PlayerDetector.filter_to_players([])
        assert result == []

    def test_max_two_returned(self):
        """Even with many detections, at most 2 should be returned."""
        dets = [
            Detection(track_id=i, bbox=(i * 100, 200, i * 100 + 50, 400),
                      confidence=0.9, foot_point=(i * 100 + 25, 400))
            for i in range(5)
        ]
        result = PlayerDetector.filter_to_players(dets, frame_shape=(720, 1080, 3))
        assert len(result) <= 2

    def test_single_detection_passes(self):
        """A single detection should be returned as-is (if it scores well)."""
        det = Detection(
            track_id=1, bbox=(400, 200, 500, 400),
            confidence=0.9, foot_point=(450.0, 400.0),
        )
        result = PlayerDetector.filter_to_players([det], frame_shape=(720, 1080, 3))
        assert len(result) == 1


# ===========================================================================
# analytics.py tests
# ===========================================================================

class TestAnalyticsEngine:
    """Tests for the post-match analytics computation."""

    def test_empty_samples(self):
        """Analytics should handle zero samples gracefully."""
        engine = AnalyticsEngine(fps=30.0, total_frames=100)
        result = engine.compute(p1_distance=0.0, p2_distance=0.0)
        assert result.player_1.avg_speed_kmh == 0.0
        assert result.player_2.court_coverage_pct == 0.0

    def test_stationary_player(self):
        """A player standing still should have near-zero average speed."""
        engine = AnalyticsEngine(fps=30.0, total_frames=100)
        for i in range(100):
            engine.record(1, i, (4.0, 12.0), speed_mps=0.0, detected=True)
        result = engine.compute(p1_distance=0.0, p2_distance=0.0)
        assert result.player_1.avg_speed_kmh == 0.0
        assert result.player_1.movement.stationary_pct > 90.0

    def test_sprinting_player(self):
        """A player moving at 5 m/s should be classified as sprinting."""
        engine = AnalyticsEngine(fps=30.0, total_frames=100)
        for i in range(100):
            x = 4.0 + (i / 100.0) * 4.0
            engine.record(1, i, (x, 12.0), speed_mps=5.0, detected=True)
        result = engine.compute(p1_distance=4.0, p2_distance=0.0)
        assert result.player_1.movement.sprinting_pct > 90.0
        assert result.player_1.avg_speed_kmh > 15.0

    def test_detection_rate(self):
        """Detection rate should reflect how many frames the player was seen."""
        engine = AnalyticsEngine(fps=30.0, total_frames=100)
        for i in range(100):
            detected = i % 2 == 0  # Detected every other frame
            engine.record(1, i, (4.0, 12.0), speed_mps=0.0, detected=detected)
        result = engine.compute(p1_distance=0.0, p2_distance=0.0)
        assert result.player_1.detection_rate_pct == pytest.approx(50.0)

    def test_court_coverage_changes_with_positions(self):
        """A player visiting many positions should have higher court coverage."""
        engine = AnalyticsEngine(fps=30.0, total_frames=200)
        # Player 1: visits many positions across the court
        for i in range(200):
            x = (i % 20) / 20.0 * config.COURT_WIDTH_M
            y = (i // 20) / 10.0 * config.COURT_LENGTH_M
            engine.record(1, i, (x, y), speed_mps=2.0, detected=True)

        # Player 2: stays in one spot
        for i in range(200):
            engine.record(2, i, (4.0, 12.0), speed_mps=0.0, detected=True)

        result = engine.compute(p1_distance=50.0, p2_distance=0.0)
        assert result.player_1.court_coverage_pct > result.player_2.court_coverage_pct

    def test_analytics_to_dict_structure(self):
        """Verify the JSON-serializable dict has the expected structure."""
        engine = AnalyticsEngine(fps=30.0, total_frames=10)
        for i in range(10):
            engine.record(1, i, (4.0, 12.0), speed_mps=1.0, detected=True)
            engine.record(2, i, (4.0, 12.0), speed_mps=1.0, detected=True)
        result = engine.compute(p1_distance=10.0, p2_distance=10.0)
        d = AnalyticsEngine.analytics_to_dict(result)

        assert "analytics" in d
        assert "player_1" in d["analytics"]
        assert "player_2" in d["analytics"]
        p1 = d["analytics"]["player_1"]
        assert "speed" in p1
        assert "movement_phases" in p1
        assert "court_zones" in p1
        assert "court_coverage_pct" in p1
        assert "detection_quality" in p1


class TestMovementPhaseThresholds:
    """Verify that movement phase thresholds are properly ordered."""

    def test_thresholds_increasing(self):
        assert STATIONARY_THRESHOLD_MPS < WALK_THRESHOLD_MPS < JOG_THRESHOLD_MPS

    def test_stationary_below_walking(self):
        assert STATIONARY_THRESHOLD_MPS < 1.0


# ===========================================================================
# config.py sanity tests
# ===========================================================================

class TestConfig:
    """Sanity checks for configuration values."""

    def test_court_dimensions_positive(self):
        assert config.COURT_WIDTH_M > 0
        assert config.COURT_LENGTH_M > 0

    def test_court_is_longer_than_wide(self):
        """A tennis court should be longer than it is wide."""
        assert config.COURT_LENGTH_M > config.COURT_WIDTH_M

    def test_itf_dimensions(self):
        """Verify ITF standard singles court dimensions."""
        assert config.COURT_WIDTH_M == pytest.approx(8.23, abs=0.01)
        assert config.COURT_LENGTH_M == pytest.approx(23.77, abs=0.01)

    def test_confidence_threshold_valid(self):
        assert 0.0 < config.CONF_THRESHOLD < 1.0

    def test_ema_alpha_valid(self):
        assert 0.0 < config.EMA_ALPHA < 1.0

    def test_min_distance_threshold_positive(self):
        assert config.MIN_DIST_THRESHOLD_M > 0

    def test_max_delta_greater_than_min(self):
        assert config.MAX_DELTA_PER_FRAME_M > config.MIN_DIST_THRESHOLD_M

    def test_device_valid(self):
        assert config.DEVICE in ("cuda", "cpu")
