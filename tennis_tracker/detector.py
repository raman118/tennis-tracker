"""
Player detection module for the Tennis Player Tracking pipeline.

Wraps YOLOv8x with ByteTrack to detect and track persons in each video
frame, then filters detections down to the two on-court players using a
multi-signal scoring system that considers court position, bounding box
geometry, size, and YOLO confidence.

Architecture decision: Detection and player-filtering are separated into
two methods (detect → filter_to_players) so that upstream code can inspect
raw detections for debugging before filtering is applied. This follows the
"progressive refinement" pattern common in CV pipelines.

The scoring system replaces naive confidence-only filtering to prevent
ball boys, chair umpires, and line judges from being confused with players.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from tennis_tracker import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single person detection in a video frame.

    Attributes:
        track_id: Persistent ID assigned by ByteTrack across frames.
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Model's confidence score for this detection [0, 1].
        foot_point: Bottom-center of the bbox — the estimated ground
                    contact point (cx, y2). Using this instead of the
                    bbox center dramatically improves homography accuracy
                    because it corresponds to the court plane, not the
                    player's torso which is elevated above the ground.
    """

    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    foot_point: Tuple[float, float]


class PlayerDetector:
    """Detects and tracks persons using YOLOv8x + ByteTrack.

    This class manages the YOLO model lifecycle and provides methods to
    run per-frame detection with persistent tracking, then filter the
    results to keep only the two on-court tennis players using a
    multi-signal scoring system.
    """

    def __init__(self) -> None:
        """Load the YOLOv8x model onto the configured device.

        The model is loaded once and reused for all frames. ByteTrack
        state is maintained internally by the ultralytics tracker when
        persist=True is passed to model.track().
        """
        logger.info("Loading YOLO model: %s on device: %s", config.YOLO_MODEL, config.DEVICE)

        # YOLO() auto-downloads the weights on first run if not cached locally
        self.model = YOLO(config.YOLO_MODEL)

        # Move model to the target device (cuda or cpu).
        # This is a no-op if the model is already on the correct device.
        self.device: str = config.DEVICE

        # Reset class-level filter call counter for this detector instance.
        # This ensures batch processing across multiple videos starts fresh.
        PlayerDetector._filter_call_count = 0

        logger.info("YOLO model loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection + tracking on a single video frame.

        Uses model.track() with persist=True which tells ByteTrack to
        maintain its internal Kalman filter state across calls, enabling
        persistent track IDs.

        Args:
            frame: BGR uint8 image from cv2.VideoCapture.

        Returns:
            List of Detection objects for all persons found in the frame.
            May include non-players (ball boys, umpire). Use
            filter_to_players() to narrow down.
        """
        # model.track() integrates detection + tracking in one call.
        # - persist=True: maintain tracker state across frames
        # - conf: minimum confidence to keep a detection
        # - classes: filter to COCO class 0 (person) at the model level,
        #   which is more efficient than post-hoc filtering
        # - verbose=False: suppress per-frame YOLO console spam
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.CONF_THRESHOLD,
            classes=[config.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        # model.track() returns a list of Results objects (one per image).
        # Since we pass a single frame, results[0] has our data.
        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Extract bounding box coordinates (xyxy format: x1, y1, x2, y2)
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                # Extract confidence score
                conf = float(boxes.conf[i].cpu().numpy())

                # Extract track ID. ByteTrack may not assign an ID immediately
                # (e.g., first frame), so we default to -1 for untracked detections.
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    track_id = -1

                # Compute the foot point: bottom-center of the bounding box.
                # This is the estimated ground contact position of the player.
                # (x1+x2)/2 gives horizontal center; y2 gives the bottom edge.
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                detections.append(
                    Detection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        foot_point=(foot_x, foot_y),
                    )
                )

        return detections

    @staticmethod
    def compute_detection_score(
        detection: Detection,
        frame_shape: Tuple[int, ...],
    ) -> float:
        """Compute a multi-signal score indicating how likely a detection is a real player.

        Combines four independent signals into a weighted total score:
          1. Court zone position (40%) — players are in the middle of the frame
          2. Bounding box aspect ratio (25%) — players have tall, narrow boxes
          3. Bounding box size (20%) — players occupy a meaningful frame fraction
          4. YOLO confidence (15%) — model's own confidence in the detection

        This replaces naive confidence-only filtering and dramatically reduces
        confusion with ball boys, chair umpires, and line judges.

        Args:
            detection: A single Detection to score.
            frame_shape: Shape of the video frame (height, width, channels).

        Returns:
            Total weighted score in [0.0, 1.0].
        """
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = detection.bbox

        # --- SIGNAL 1: Court Zone Score (weight: 40%) ---
        # Where in the frame is the detection's foot point?
        # Players live in the center of the frame; sideline officials are at edges.
        foot_x, foot_y = detection.foot_point
        foot_y_frac = foot_y / max(frame_h, 1)
        foot_x_frac = foot_x / max(frame_w, 1)

        # Horizontal zone (based on vertical position in frame)
        if foot_y_frac < config.COURT_H_TOP:
            h_zone_score = 0.2    # far baseline — valid but distant
        elif foot_y_frac > config.COURT_H_BOTTOM:
            h_zone_score = 0.8    # near baseline — valid player zone
        else:
            h_zone_score = 1.0    # main court — highest probability

        # Vertical zone (based on horizontal position in frame)
        if foot_x_frac < config.COURT_V_LEFT:
            v_zone_score = 0.1    # sideline chair, crowd
        elif foot_x_frac > config.COURT_V_RIGHT:
            v_zone_score = 0.1    # sideline chair, crowd
        else:
            v_zone_score = 1.0    # court center — players live here

        court_zone_score = h_zone_score * v_zone_score

        # --- SIGNAL 2: Bounding Box Aspect Ratio Score (weight: 25%) ---
        # Real players have tall, narrow bounding boxes (height/width ~ 1.5–3.5).
        bbox_w = max(x2 - x1, 1.0)
        bbox_h = max(y2 - y1, 1.0)
        ratio = bbox_h / bbox_w

        if 1.5 <= ratio <= 3.5:
            aspect_score = 1.0     # ideal player bbox shape
        elif 1.0 <= ratio < 1.5:
            aspect_score = 0.5     # slightly wide — could be mid-swing
        elif ratio > 3.5:
            aspect_score = 0.3     # too tall — likely a pole or umpire stand
        else:
            aspect_score = 0.0     # wider than tall — almost never a player

        # --- SIGNAL 3: Bounding Box Size Score (weight: 20%) ---
        # Players occupy a meaningful fraction of frame height.
        # Ball boys and distant people are much smaller.
        height_fraction = bbox_h / max(frame_h, 1)

        if 0.12 <= height_fraction <= 0.55:
            size_score = 1.0       # normal player size
        elif 0.08 <= height_fraction < 0.12:
            size_score = 0.5       # small — possibly far baseline player
        elif height_fraction > 0.55:
            size_score = 0.2       # too large — camera artifact or close-up
        else:
            size_score = 0.0       # too small — ball boy or child

        # --- SIGNAL 4: Detection Confidence Score (weight: 15%) ---
        # Raw YOLO confidence, already normalized to [0.0, 1.0].
        confidence_score = detection.confidence

        # --- FINAL WEIGHTED SCORE ---
        total = (
            config.W_COURT_ZONE * court_zone_score
            + config.W_ASPECT * aspect_score
            + config.W_SIZE * size_score
            + config.W_CONFIDENCE * confidence_score
        )

        return total

    # Class-level frame counter for periodic logging in filter_to_players.
    # Reset in __init__ so batch processing across videos starts fresh.
    _filter_call_count: int = 0

    @staticmethod
    def filter_to_players(
        detections: List[Detection],
        calibrator: Optional[object] = None,
        locked_track_ids: Optional[Tuple[Optional[int], Optional[int]]] = None,
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> List[Detection]:
        """Filter detections to keep only the two on-court tennis players.

        Uses a multi-signal scoring system to rank every detection, then
        keeps the top 2 by total score. This replaces the old confidence-only
        approach which was easily fooled by ball boys and umpires.

        Filtering cascade:
          1. If a calibrator with court polygon is available, keep only
             detections whose foot_point is inside the court boundaries.
          2. If locked_track_ids are provided (post-warmup), ALWAYS keep
             detections matching those IDs regardless of score.
             Fill remaining slots with highest-scoring detections.
          3. If no locked IDs, score all detections and keep top 2 above
             the minimum threshold of 0.25.

        Args:
            detections: All person detections in the current frame.
            calibrator: Optional CourtCalibrator instance with court polygon.
            locked_track_ids: Tuple of (p1_locked_id, p2_locked_id) if tracking
                              has locked player identities, None otherwise.
            frame_shape: Shape of the current frame (height, width, channels).
                         Required for detection scoring. If None, falls back
                         to confidence-only filtering.

        Returns:
            At most 2 Detection objects representing the on-court players.
        """
        if not detections:
            return []

        # Step 1: Court polygon filtering (if calibrator is available)
        if calibrator is not None and hasattr(calibrator, "is_inside_court"):
            court_detections = [
                d for d in detections if calibrator.is_inside_court(d.foot_point)
            ]
            # Only use the filtered list if it has enough players.
            # If court polygon is too tight and eliminates real players,
            # fall through to scoring-based selection on the full set.
            if len(court_detections) >= 1:
                detections = court_detections

        # Step 2: Multi-signal scoring (if frame_shape is available)
        if frame_shape is not None:
            # Score every detection
            scored: List[Tuple[float, Detection]] = []
            for det in detections:
                score = PlayerDetector.compute_detection_score(det, frame_shape)
                scored.append((score, det))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Periodic logging: log top-3 scores every 30 frames
            PlayerDetector._filter_call_count += 1
            if PlayerDetector._filter_call_count % 30 == 0:
                top_n = scored[:3]
                for rank, (sc, det) in enumerate(top_n, 1):
                    logger.info(
                        "Frame %d | Detection rank #%d: track_id=%d, "
                        "score=%.3f, conf=%.2f, foot=(%.0f, %.0f), "
                        "bbox_size=%.0fx%.0f",
                        PlayerDetector._filter_call_count,
                        rank,
                        det.track_id,
                        sc,
                        det.confidence,
                        det.foot_point[0],
                        det.foot_point[1],
                        det.bbox[2] - det.bbox[0],
                        det.bbox[3] - det.bbox[1],
                    )

            # Post-warmup: prioritize locked IDs, fill remaining with top scores
            if locked_track_ids is not None:
                locked_dets = [
                    det for det in detections
                    if det.track_id in locked_track_ids and det.track_id >= 0
                ]
                unlocked_scored = [
                    (sc, det) for sc, det in scored
                    if det.track_id not in locked_track_ids or det.track_id < 0
                ]

                remaining_slots = config.MAX_PLAYERS - len(locked_dets)
                if remaining_slots > 0:
                    # Fill with highest-scoring non-locked detections
                    for sc, det in unlocked_scored[:remaining_slots]:
                        if sc >= 0.25:
                            locked_dets.append(det)

                return locked_dets[:config.MAX_PLAYERS]

            # Pre-warmup: keep top 2 above minimum score threshold
            result: List[Detection] = []
            for sc, det in scored:
                if len(result) >= config.MAX_PLAYERS:
                    break
                if sc >= 0.25:
                    result.append(det)
            return result

        # Fallback: if frame_shape is not available, use old confidence logic
        # (This should not happen in normal operation)
        if locked_track_ids is not None and len(detections) > config.MAX_PLAYERS:
            locked_dets = [
                d for d in detections
                if d.track_id in locked_track_ids and d.track_id >= 0
            ]
            unlocked_dets = [
                d for d in detections
                if d.track_id not in locked_track_ids or d.track_id < 0
            ]

            remaining_slots = config.MAX_PLAYERS - len(locked_dets)
            if remaining_slots > 0:
                unlocked_dets = sorted(
                    unlocked_dets, key=lambda d: d.confidence, reverse=True
                )
                locked_dets.extend(unlocked_dets[:remaining_slots])

            return locked_dets[:config.MAX_PLAYERS]

        if len(detections) > config.MAX_PLAYERS:
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
            detections = detections[: config.MAX_PLAYERS]

        return detections
