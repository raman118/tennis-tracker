"""
Player tracking and distance accumulation module.

Maintains persistent identity for two tennis players across all video frames,
smooths their positions via adaptive EMA, transforms to real-world meters via
homography, and accumulates total distance traveled.

Improvements over baseline:
  - Scored warmup: locks IDs based on cumulative detection scores, not just
    frequency, so a high-scoring player beats a frequently-seen ball boy.
  - Active re-locking: when a locked ID goes stale (player lost for >5 frames),
    the tracker actively re-locks to the best-scoring unmatched detection.
    This is inspired by the tennis_analysis reference repo's approach of
    re-selecting players every frame rather than permanently freezing IDs.
  - Velocity-gated re-ID: prevents far-away detections from being force-
    assigned to the wrong player after an occlusion.
  - Temporal consistency: holds player state for up to 5 missed frames
    instead of immediately losing tracking continuity.
  - Adaptive EMA alpha: responds faster during sprints, smooths harder
    when the player is stationary.
  - Distance sanity cap: discards single-frame spikes that exceed the
    physically possible displacement for a tennis player.

Architecture decision: tracking state is separated from detection so that
the tracker can handle missing detections (held state) and re-ID failures
(nearest-centroid recovery) independently of the detector's output format.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from tennis_tracker import config
from tennis_tracker.calibration import CourtCalibrator
from tennis_tracker.detector import Detection, PlayerDetector
from tennis_tracker.utils import ema_smooth, euclidean_dist

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """Mutable state for a single tracked player.

    Encapsulates everything needed to render overlays and compute distances
    for one player across the entire video.

    Attributes:
        player_id: Logical player index (1 or 2) — stable across the video.
        locked_track_id: ByteTrack ID that was locked to this player after
                         warmup. May be re-locked if the ID goes stale.
        bbox: Most recent bounding box (x1, y1, x2, y2) for rendering.
        smoothed_foot: EMA-smoothed foot position in pixel space.
        meter_position: Current position in real-world meter space.
        prev_meter_position: Previous frame's meter position for delta calc.
        total_distance_m: Cumulative distance traveled in meters.
        trail: Fixed-length deque of recent smoothed foot positions for
               drawing the fading trajectory polyline.
        last_confidence: Most recent detection confidence for this player.
        detected_this_frame: Whether this player was detected in the
                             current frame (used to decide whether to update).
        missed_frames: Number of consecutive frames where this player was
                       not detected. Used for temporal consistency — up to 5
                       missed frames are tolerated before triggering re-lock.
        last_detection_score: Most recent multi-signal detection score for
                              this player. Used to render the debug score bar
                              in the visualizer overlay.
        prev_meter_pos: Previous frame's meter position for computing
                        instantaneous speed. Unlike prev_meter_position
                        (which is updated only on distance accumulation),
                        this is updated every frame for speed display.
    """

    player_id: int
    locked_track_id: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    smoothed_foot: Optional[Tuple[float, float]] = None
    meter_position: Optional[Tuple[float, float]] = None
    prev_meter_position: Optional[Tuple[float, float]] = None
    total_distance_m: float = 0.0
    trail: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH)
    )
    last_confidence: float = 0.0
    detected_this_frame: bool = False
    missed_frames: int = 0
    last_detection_score: float = 0.0
    prev_meter_pos: Optional[Tuple[float, float]] = None


class PlayerTracker:
    """Tracks two tennis players with stable identities and distance accumulation.

    Lifecycle:
      1. WARMUP PHASE (frames 0 to WARMUP_FRAMES-1):
         Observe all track IDs and accumulate their detection scores.
         No distance is accumulated because IDs are not yet stable.

      2. ID LOCKING (frame == WARMUP_FRAMES):
         Lock the two track IDs with the HIGHEST CUMULATIVE SCORES as
         Player 1 and Player 2. Score-based locking ensures a player who
         consistently scores 0.9 beats a ball boy who scores 0.2.

      3. TRACKING PHASE (frames > WARMUP_FRAMES):
         Match incoming detections to locked IDs. Handle re-ID failures via
         velocity-gated nearest-centroid assignment. If a player's locked ID
         goes stale (missed >5 frames), actively re-lock to the best available
         unmatched detection. Smooth positions with adaptive EMA, transform
         to meters, accumulate distance with sanity cap.
    """

    # Threshold: after this many consecutive missed frames, aggressively re-lock
    # to any available high-scoring detection instead of waiting for the original ID
    RELOCK_AFTER_FRAMES: int = 5

    def __init__(self, calibrator: CourtCalibrator) -> None:
        """Initialize the tracker with two empty player states.

        Args:
            calibrator: Court calibrator for pixel-to-meter transforms.
        """
        self.calibrator = calibrator

        # Player state objects — one per player
        self.p1 = PlayerState(player_id=1)
        self.p2 = PlayerState(player_id=2)

        # Warmup: accumulate detection scores per track ID instead of raw
        # frequency counts. This ensures high-quality detections (real players)
        # outweigh low-quality detections (ball boys) even if the ball boy
        # appears in more frames.
        self._track_scores: Dict[int, float] = {}

        # Flag: have we locked player IDs yet?
        self._ids_locked: bool = False

        # Track the warmup detections so we can initialize positions at lock time
        self._warmup_detections: Dict[int, Detection] = {}

        # Store frame shape for detection scoring during warmup
        self._frame_shape: Optional[Tuple[int, ...]] = None

        logger.info(
            "PlayerTracker initialized. Warmup phase: %d frames.", config.WARMUP_FRAMES
        )

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Process one frame's detections and update player states.

        This is the main entry point called once per video frame. It handles
        the warmup → locking → tracking state machine, including temporal
        consistency for missed detections and active re-locking when IDs
        go stale.

        Args:
            detections: Filtered person detections for this frame (≤ 2).
            frame_idx: Zero-based frame index.
            frame_shape: Shape of the current frame (height, width, channels).
                         Needed for detection scoring during warmup.
        """
        # Store frame shape for warmup scoring
        if frame_shape is not None:
            self._frame_shape = frame_shape

        # Reset detection flags — will be set to True if matched this frame
        self.p1.detected_this_frame = False
        self.p2.detected_this_frame = False

        if not self._ids_locked:
            self._warmup_phase(detections, frame_idx)
        else:
            self._tracking_phase(detections)

            # --- Temporal consistency + active re-locking ---
            # Handle players that were NOT matched this frame
            for player in (self.p1, self.p2):
                if not player.detected_this_frame:
                    player.missed_frames += 1
                    if player.missed_frames <= self.RELOCK_AFTER_FRAMES:
                        # Hold last position — don't update distance, but keep
                        # trail alive. The player is probably just temporarily
                        # occluded or the detector missed them for a frame.
                        pass
                    elif player.missed_frames == self.RELOCK_AFTER_FRAMES + 1:
                        # Log once at the transition point (not every frame)
                        logger.warning(
                            "P%d lost for %d frames — will re-lock on next match.",
                            player.player_id,
                            player.missed_frames,
                        )
                    # NOTE: smoothed_foot is NOT reset to None anymore.
                    # Instead, we keep it around so re-locking can check
                    # whether the new detection is spatially reasonable.
                    # The active re-locking in _tracking_phase handles recovery.
                else:
                    # Successfully detected — reset the missed frames counter
                    player.missed_frames = 0

    def _warmup_phase(
        self, detections: List[Detection], frame_idx: int
    ) -> None:
        """Collect track ID statistics during the warmup window.

        Instead of just counting appearances (which can be fooled by a ball boy
        that is on-screen more than a player), we accumulate the detection score
        for each track ID. A player who scores 0.9 every frame will have a much
        higher cumulative score than a ball boy who scores 0.2 every frame.

        Args:
            detections: Detections for the current frame.
            frame_idx: Zero-based frame index.
        """
        for det in detections:
            if det.track_id >= 0:
                # Compute detection score if frame shape is available
                if self._frame_shape is not None:
                    score = PlayerDetector.compute_detection_score(
                        det, self._frame_shape
                    )
                else:
                    # Fallback: use confidence as score if frame shape unknown
                    score = det.confidence

                # Accumulate score for this track ID
                if det.track_id not in self._track_scores:
                    self._track_scores[det.track_id] = 0.0
                self._track_scores[det.track_id] += score

                # Store the latest detection for each ID so we can initialize
                # positions when we lock
                self._warmup_detections[det.track_id] = det

        # Check if warmup period is complete
        if frame_idx >= config.WARMUP_FRAMES - 1:
            self._lock_ids()

    def _lock_ids(self) -> None:
        """Lock the two track IDs with the highest cumulative detection scores.

        After warmup, the two IDs with the highest cumulative scores are almost
        certainly the two on-court players. Score-based locking is superior to
        frequency-based locking because it weights detection quality: a player
        who appears in 20 frames with score 0.9 each (total=18.0) outranks a
        ball boy who appears in 30 frames with score 0.2 each (total=6.0).
        """
        if len(self._track_scores) < 2:
            logger.warning(
                "Only %d unique track IDs seen during warmup (need 2). "
                "Will attempt to lock once more IDs appear.",
                len(self._track_scores),
            )
            # Don't lock yet — extend warmup implicitly
            return

        # Sort track IDs by cumulative score descending, take top 2
        sorted_ids = sorted(
            self._track_scores.items(), key=lambda x: x[1], reverse=True
        )
        id_1, score_1 = sorted_ids[0]
        id_2, score_2 = sorted_ids[1]

        self.p1.locked_track_id = id_1
        self.p2.locked_track_id = id_2

        # Initialize player positions from their latest warmup detections
        if id_1 in self._warmup_detections:
            det = self._warmup_detections[id_1]
            self.p1.smoothed_foot = det.foot_point
            self.p1.meter_position = self.calibrator.transform(det.foot_point)
            self.p1.prev_meter_position = self.p1.meter_position
            self.p1.prev_meter_pos = self.p1.meter_position
            self.p1.bbox = det.bbox

        if id_2 in self._warmup_detections:
            det = self._warmup_detections[id_2]
            self.p2.smoothed_foot = det.foot_point
            self.p2.meter_position = self.calibrator.transform(det.foot_point)
            self.p2.prev_meter_position = self.p2.meter_position
            self.p2.prev_meter_pos = self.p2.meter_position
            self.p2.bbox = det.bbox

        self._ids_locked = True

        logger.info(
            "Player IDs locked after warmup — P1: track_id=%d (cumulative_score=%.2f), "
            "P2: track_id=%d (cumulative_score=%.2f).",
            id_1, score_1, id_2, score_2,
        )

        # Log all tracked IDs for debugging
        if len(sorted_ids) > 2:
            for tid, sc in sorted_ids[2:]:
                logger.debug(
                    "  Rejected track_id=%d (cumulative_score=%.2f) — likely non-player.",
                    tid, sc,
                )

        # Free warmup memory — no longer needed
        self._warmup_detections.clear()
        self._track_scores.clear()

    def _tracking_phase(self, detections: List[Detection]) -> None:
        """Match detections to locked player IDs and update states.

        Matching strategy (3-tier):
          1. Direct ID match: if a detection's track_id matches a locked ID,
             assign it directly.
          2. Nearest-centroid recovery: if ByteTrack assigned a new ID after
             an occlusion, assign based on spatial proximity with velocity gate.
          3. Active re-locking: if a player has been lost for >5 frames and
             there are still unmatched high-scoring detections, re-lock the
             player's ID to the best available detection. This is the key fix
             inspired by the tennis_analysis reference repo's approach of
             re-selecting players every frame.

        Args:
            detections: Filtered detections for this frame (≤ 2).
        """
        # Build a lookup of which detections match which tracked player
        p1_det: Optional[Detection] = None
        p2_det: Optional[Detection] = None
        unmatched: List[Detection] = []

        for det in detections:
            if det.track_id == self.p1.locked_track_id:
                p1_det = det
            elif det.track_id == self.p2.locked_track_id:
                p2_det = det
            else:
                # This detection has a track_id we don't recognize — likely a
                # re-ID failure where ByteTrack lost the original ID after an
                # occlusion or camera artifact.
                unmatched.append(det)

        # --- Tier 2: Nearest-centroid recovery for unmatched detections ---
        # Only applies when smoothed_foot is still valid (player not yet lost)
        still_unmatched: List[Detection] = []
        for det in unmatched:
            assigned = self._nearest_centroid_assign(det, p1_det, p2_det)
            if assigned == 1:
                p1_det = det
            elif assigned == 2:
                p2_det = det
            else:
                still_unmatched.append(det)

        # --- Tier 3: Active re-locking for lost players ---
        # If a player has been lost for too long and there are unmatched
        # detections available, re-lock the player to the best-scoring one.
        # This prevents the "permanent loss" bug where locked IDs go stale
        # and the player can never be re-acquired.
        # NOTE: _try_relock() fully updates player state internally (foot,
        # bbox, meter_position, trail, etc.), so we must NOT call
        # _update_player() again for re-locked players — that would cause
        # double trail entries and redundant EMA smoothing.
        if still_unmatched:
            self._try_relock(still_unmatched, p1_det, p2_det)

        # Update each player's state if they were detected this frame
        # via Tier 1 (direct ID match) or Tier 2 (nearest-centroid).
        # Players updated via Tier 3 (re-lock) are already handled inside
        # _try_relock and have detected_this_frame=True, so we skip them
        # here to avoid double-updating.
        if p1_det is not None and not self.p1.detected_this_frame:
            self._update_player(self.p1, p1_det)
        if p2_det is not None and not self.p2.detected_this_frame:
            self._update_player(self.p2, p2_det)

    def _try_relock(
        self,
        unmatched: List[Detection],
        p1_det: Optional[Detection],
        p2_det: Optional[Detection],
    ) -> None:
        """Attempt to re-lock lost players to the best available unmatched detections.

        This is the critical fix for the permanent player loss bug. When a player's
        locked track_id goes stale (ByteTrack reassigns it), the player accumulates
        missed_frames. After RELOCK_AFTER_FRAMES consecutive misses, we actively
        re-lock to the best-scoring available detection instead of waiting forever
        for the original (now-dead) track ID.

        The detection is scored using the multi-signal scoring system, and only
        detections with score >= 0.3 are eligible for re-locking. The player that
        has been lost the longest is re-locked first.

        Args:
            unmatched: Detections that didn't match any locked ID and weren't
                       recovered by nearest-centroid.
            p1_det: Currently matched P1 detection, or None.
            p2_det: Currently matched P2 detection, or None.
        """
        if not unmatched:
            return

        # Score unmatched detections
        scored_unmatched: List[Tuple[float, Detection]] = []
        for det in unmatched:
            if self._frame_shape is not None:
                score = PlayerDetector.compute_detection_score(det, self._frame_shape)
            else:
                score = det.confidence
            if score >= 0.3:  # Only consider plausible-player detections
                scored_unmatched.append((score, det))

        if not scored_unmatched:
            return

        # Sort by score descending
        scored_unmatched.sort(key=lambda x: x[0], reverse=True)

        # Determine which players need re-locking (lost for too long, no match this frame)
        candidates: List[PlayerState] = []
        if p1_det is None and self.p1.missed_frames > self.RELOCK_AFTER_FRAMES:
            candidates.append(self.p1)
        if p2_det is None and self.p2.missed_frames > self.RELOCK_AFTER_FRAMES:
            candidates.append(self.p2)

        if not candidates:
            return

        # Re-lock: prioritize the player that's been lost the longest
        candidates.sort(key=lambda p: p.missed_frames, reverse=True)

        used_det_ids = set()
        for player in candidates:
            for score, det in scored_unmatched:
                if id(det) in used_det_ids:
                    continue

                old_id = player.locked_track_id
                player.locked_track_id = det.track_id

                logger.info(
                    "P%d RE-LOCKED: track_id %s → %d (score=%.3f, "
                    "was lost for %d frames).",
                    player.player_id,
                    old_id,
                    det.track_id,
                    score,
                    player.missed_frames,
                )

                # Cold-start: initialize position from this detection
                # Don't reset distance — keep accumulated total
                player.smoothed_foot = det.foot_point
                player.bbox = det.bbox
                player.detected_this_frame = True
                player.last_confidence = det.confidence
                player.missed_frames = 0

                # Store detection score
                player.last_detection_score = score

                # Transform to meter-space for distance tracking
                current_meter = self.calibrator.transform(det.foot_point)
                # Reset BOTH meter reference points to the new position
                # since we jumped discontinuously. This prevents:
                # 1. False distance accumulation from old→new position
                # 2. Phantom speed spike in the visualizer
                player.prev_meter_position = current_meter
                player.prev_meter_pos = current_meter
                player.meter_position = current_meter

                # Add to trail
                player.trail.append(player.smoothed_foot)

                used_det_ids.add(id(det))
                break  # Move to next player

    def _nearest_centroid_assign(
        self,
        det: Detection,
        p1_det: Optional[Detection],
        p2_det: Optional[Detection],
    ) -> int:
        """Assign an unmatched detection to the closest unassigned player.

        Includes TWO distance gates:
          1. MAX_PIXELS_PER_FRAME (velocity gate): a real player cannot move
             more than ~80px between consecutive frames at broadcast resolution.
             This prevents assigning a detection from the far side of the court
             to the wrong player.
          2. MAX_RECOVERY_DIST_PX (existing): if the closest player is further
             than this threshold, the detection is rejected entirely to prevent
             non-players from contaminating a tracked player's position history.

        If no player passes both gates, the detection is skipped (it may be
        picked up by active re-locking instead).

        Args:
            det: The unmatched detection to assign.
            p1_det: Already-matched P1 detection (None if P1 is unassigned).
            p2_det: Already-matched P2 detection (None if P2 is unassigned).

        Returns:
            1 if assigned to P1, 2 if assigned to P2, 0 if rejected.
        """
        # Only assign to players that don't already have a detection this frame
        # AND still have a valid smoothed_foot (not yet fully lost)
        p1_available = (
            p1_det is None
            and self.p1.smoothed_foot is not None
            and self.p1.missed_frames <= self.RELOCK_AFTER_FRAMES
        )
        p2_available = (
            p2_det is None
            and self.p2.smoothed_foot is not None
            and self.p2.missed_frames <= self.RELOCK_AFTER_FRAMES
        )

        if not p1_available and not p2_available:
            # Both players already have detections or are lost (handled by re-lock)
            return 0

        # Compute distances to available players
        dist_to_p1 = (
            euclidean_dist(det.foot_point, self.p1.smoothed_foot)
            if p1_available
            else float("inf")
        )
        dist_to_p2 = (
            euclidean_dist(det.foot_point, self.p2.smoothed_foot)
            if p2_available
            else float("inf")
        )

        # --- Velocity gate (Improvement 2B) ---
        # Apply MAX_PIXELS_PER_FRAME gate: reject candidates that are
        # impossibly far from the player's last known position.
        if p1_available and dist_to_p1 > config.MAX_PIXELS_PER_FRAME:
            dist_to_p1 = float("inf")

        if p2_available and dist_to_p2 > config.MAX_PIXELS_PER_FRAME:
            dist_to_p2 = float("inf")

        # Pick the closer player
        min_dist = min(dist_to_p1, dist_to_p2)

        # If no player passed the velocity gate, skip (re-lock will handle)
        if min_dist == float("inf"):
            return 0

        # Reject assignment if the closest player is too far away
        if min_dist > config.MAX_RECOVERY_DIST_PX:
            return 0

        if dist_to_p1 <= dist_to_p2:
            return 1
        else:
            return 2

    def _update_player(self, player: PlayerState, det: Detection) -> None:
        """Update a single player's state with a new detection.

        Steps:
          1. Compute adaptive EMA alpha based on displacement (fast movement
             gets higher alpha for responsiveness, stationary gets low alpha
             for jitter suppression).
          2. Apply EMA smoothing to the foot point (in pixel space).
          3. Transform smoothed position to meter-space via homography.
          4. Compute Euclidean delta to previous meter position.
          5. Apply distance sanity cap — discard deltas above 1.5m/frame.
          6. Accumulate distance if delta exceeds noise threshold.
          7. Update speed-tracking fields and append to trajectory trail.

        Smoothing is applied in pixel space BEFORE the homography transform
        because the EMA operates on the raw detector output, which has
        pixel-level jitter. Smoothing after transform would amplify noise
        in regions where homography stretching is large.

        Args:
            player: The PlayerState to update.
            det: The detection matched to this player.
        """
        player.detected_this_frame = True
        player.bbox = det.bbox
        player.last_confidence = det.confidence

        # Store detection score for visual debug overlay
        if self._frame_shape is not None:
            player.last_detection_score = PlayerDetector.compute_detection_score(
                det, self._frame_shape
            )
        else:
            player.last_detection_score = det.confidence

        # --- Step 1: Compute adaptive EMA alpha (Improvement 2D) ---
        # Fixed alpha=0.3 is too sluggish during sprints and too noisy when
        # standing still. Adapt based on raw displacement magnitude.
        if player.smoothed_foot is not None:
            raw_displacement = euclidean_dist(det.foot_point, player.smoothed_foot)

            if raw_displacement > 30:
                # Fast movement detected — respond faster with less smoothing
                alpha = 0.6
            elif raw_displacement > 10:
                # Medium movement — moderate smoothing
                alpha = 0.4
            else:
                # Nearly stationary — smooth harder to kill jitter
                alpha = 0.15
        else:
            # First detection — use default alpha
            alpha = config.EMA_ALPHA

        # --- Step 2: EMA smooth the foot position ---
        if player.smoothed_foot is None:
            # First detection for this player — no history to smooth against
            player.smoothed_foot = det.foot_point
        else:
            player.smoothed_foot = ema_smooth(
                player.smoothed_foot, det.foot_point, alpha
            )

        # Append smoothed foot to the trajectory trail for visualization
        player.trail.append(player.smoothed_foot)

        # --- Step 3: Transform to meter-space ---
        current_meter = self.calibrator.transform(player.smoothed_foot)

        # --- Step 4 & 5: Compute delta, apply sanity cap, accumulate ---
        if player.prev_meter_position is not None:
            delta = euclidean_dist(current_meter, player.prev_meter_position)

            # Distance sanity cap (Improvement 2E):
            # A player physically cannot move more than ~1.5m in one frame at 30fps.
            # Larger deltas indicate a detection jump (ID swap, re-ID failure).
            if delta > config.MAX_DELTA_PER_FRAME_M:
                logger.debug(
                    "P%d: delta %.2fm capped — likely detection jump, "
                    "discarding this frame's distance contribution.",
                    player.player_id, delta,
                )
                delta = 0.0  # Discard this frame's contribution entirely
                # Do NOT skip updating meter_pos — just don't count the jump

            # Only accumulate if the displacement exceeds the noise floor
            # AND the detection has sufficient confidence. This prevents
            # phantom distance from jitter and unreliable detections.
            if delta > config.MIN_DIST_THRESHOLD_M and det.confidence > config.CONF_THRESHOLD:
                player.total_distance_m += delta
                # CRITICAL: Only update prev_meter_position when distance is
                # accumulated. If we updated every frame regardless, slow
                # real movement (e.g., 0.03m/frame for 10 frames = 0.3m)
                # would never cross the per-frame threshold and would be
                # silently discarded. By holding the reference point, small
                # deltas build up until the cumulative displacement from the
                # last checkpoint exceeds the threshold.
                player.prev_meter_position = current_meter
        else:
            # First time we have a meter position — set the reference point
            player.prev_meter_position = current_meter

        # --- Step 6: Update speed-tracking and current position ---
        # prev_meter_pos is updated EVERY frame (unlike prev_meter_position
        # which is only updated on accumulation) so the speed indicator
        # shows instantaneous frame-to-frame velocity.
        player.prev_meter_pos = player.meter_position
        player.meter_position = current_meter

    def get_states(self) -> Tuple[PlayerState, PlayerState]:
        """Return the current state of both players.

        Returns:
            Tuple of (P1_state, P2_state).
        """
        return self.p1, self.p2

    @property
    def ids_locked(self) -> bool:
        """Whether player IDs have been locked after warmup."""
        return self._ids_locked
