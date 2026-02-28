"""
Visualization module for the Tennis Player Tracking pipeline.

Renders all overlays onto each video frame: trajectory trails, bounding boxes,
player labels, foot-point dots, debug score bars, missed-frame indicators,
speed indicators, and the HUD panel.

Architecture decision: all draw methods mutate the frame in-place for memory
efficiency (avoiding a copy-per-layer). The render() method orchestrates the
correct draw order: trails → boxes → debug overlays → labels → dots → HUD.
This ensures that text is always on top and trails are behind players.

Debug overlays (score bar, missed-frame indicator, speed label) help verify
that the multi-signal scoring and temporal consistency systems are working
correctly by making internal tracking state visible on every frame.
"""

import logging
from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np

from tennis_tracker import config
from tennis_tracker.mini_court import MiniCourt
from tennis_tracker.tracker import PlayerState
from tennis_tracker.utils import euclidean_dist, format_time

logger = logging.getLogger(__name__)


class Visualizer:
    """Renders all visual overlays onto video frames.

    Draw order (back-to-front):
      1. Trajectory trails (fading polylines — drawn first so they sit behind everything)
      2. Bounding boxes (around each player)
      3. Debug overlays: score bar, missed-frame indicator, speed label
      4. Player labels ("P1: 47.23 m" — above each bbox)
      5. Foot-point dots (confirms the tracking anchor is at the feet)
      6. HUD panel (semi-transparent summary stats in top-left corner)
    """

    def __init__(self) -> None:
        """Initialize the visualizer.

        Creates a MiniCourt renderer for the bird's-eye view overlay.
        """
        self._mini_court = MiniCourt()
        logger.info("Visualizer initialized.")

    def draw_trail(
        self,
        frame: np.ndarray,
        trail: Deque[Tuple[float, float]],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a fading trajectory polyline from a player's recent foot positions.

        The trail fades linearly from 10% opacity (oldest point) to 100%
        opacity (newest point). To avoid copying the entire frame per segment
        (which is extremely expensive at ~60 copies/player/frame), we batch
        all segments into a single overlay and use the average alpha.

        Args:
            frame: The video frame to draw on (mutated in-place).
            trail: Deque of (x, y) smoothed foot positions.
            color: BGR color tuple for the trail.
        """
        points = list(trail)
        n = len(points)

        if n < 2:
            # Need at least 2 points to draw a line segment
            return

        # Batch approach: draw ALL segments directly on the frame
        # using brightness-modulated color to simulate per-segment fading.
        # This reduces the per-player cost from O(n) frame copies to O(1).

        for i in range(n - 1):
            # Linear interpolation: oldest segment gets 10% alpha, newest gets 100%
            alpha_frac = 0.1 + 0.9 * (i / max(n - 1, 1))

            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))

            # Simulate fading by adjusting the color brightness per segment
            # rather than doing a separate alpha blend per segment.
            faded_color = (
                int(color[0] * alpha_frac),
                int(color[1] * alpha_frac),
                int(color[2] * alpha_frac),
            )
            cv2.line(frame, pt1, pt2, faded_color, thickness=2, lineType=cv2.LINE_AA)

    def draw_score_bar(
        self,
        frame: np.ndarray,
        state: PlayerState,
    ) -> None:
        """Draw a thin horizontal score bar above the player's bounding box.

        The bar visually represents the player's most recent detection score
        (0.0–1.0) so it's instantly visible if a detection is borderline.
        Color coding: green if score > 0.7, yellow if 0.4–0.7, red if < 0.4.

        Args:
            frame: The video frame to draw on (mutated in-place).
            state: Current PlayerState containing last_detection_score and bbox.
        """
        if state.bbox is None:
            return

        x1, y1, x2, y2 = state.bbox
        x1, y1 = int(x1), int(y1)

        # Bar dimensions: 60px wide, 6px tall, positioned above the bbox
        bar_width = 60
        bar_height = 6
        bar_x = x1
        bar_y = y1 - bar_height - 2  # 2px gap above bbox top edge

        # Ensure bar doesn't go above the frame
        if bar_y < 0:
            bar_y = int(y2) + 2  # Place below bbox instead

        # Determine fill proportion and color based on score
        score = state.last_detection_score
        fill_width = int(bar_width * min(max(score, 0.0), 1.0))

        if score > 0.7:
            bar_color = (0, 200, 0)      # Green — strong detection
        elif score >= 0.4:
            bar_color = (0, 200, 200)    # Yellow (BGR) — borderline
        else:
            bar_color = (0, 0, 200)      # Red — weak detection

        # Draw background (dark gray) for the full bar width
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (60, 60, 60),
            -1,
        )

        # Draw filled portion in the score color
        if fill_width > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                bar_color,
                -1,
            )

        # Draw thin border around the bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (180, 180, 180),
            1,
        )

    def draw_missed_frame_indicator(
        self,
        frame: np.ndarray,
        state: PlayerState,
    ) -> None:
        """Draw a dashed border and 'SEARCHING...' label when a player is temporarily lost.

        When player.missed_frames > 0, this draws a dashed rectangle where
        the player was last seen and shows how many frames have been missed.
        This makes it immediately clear when the tracker has temporarily lost
        someone and is holding their last known position.

        Args:
            frame: The video frame to draw on (mutated in-place).
            state: Current PlayerState containing missed_frames and bbox.
        """
        if state.bbox is None or state.missed_frames <= 0:
            return

        x1, y1, x2, y2 = state.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a dashed border by drawing short line segments along the bbox edges
        dash_length = 8
        gap_length = 5
        color = (0, 165, 255)  # Orange (BGR)

        # Helper to draw dashed line between two points
        edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]

        for (sx, sy), (ex, ey) in edges:
            dx = ex - sx
            dy = ey - sy
            length = max(int(np.sqrt(dx * dx + dy * dy)), 1)
            num_segments = length // (dash_length + gap_length) + 1

            for seg in range(num_segments):
                start_frac = seg * (dash_length + gap_length) / length
                end_frac = min((seg * (dash_length + gap_length) + dash_length) / length, 1.0)

                pt1 = (int(sx + dx * start_frac), int(sy + dy * start_frac))
                pt2 = (int(sx + dx * end_frac), int(sy + dy * end_frac))

                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw "SEARCHING... (Nf)" label above the box
        label = f"SEARCHING... ({state.missed_frames}f)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        label_x = x1
        label_y = y1 - 14

        if label_y < 15:
            label_y = y2 + 18

        # Shadow for contrast
        cv2.putText(
            frame, label, (label_x + 1, label_y + 1),
            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        # Orange foreground text
        cv2.putText(
            frame, label, (label_x, label_y),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )

    def draw_speed_indicator(
        self,
        frame: np.ndarray,
        state: PlayerState,
        color: Tuple[int, int, int],
        fps: float,
    ) -> None:
        """Draw the player's estimated instantaneous speed below their label.

        Computes speed from the frame-to-frame meter displacement × fps,
        then converts to km/h for intuitive display. This helps verify that
        distance accumulation is producing reasonable values.

        Args:
            frame: The video frame to draw on (mutated in-place).
            state: Current PlayerState with meter_position and prev_meter_pos.
            color: BGR color tuple for the text.
            fps: Video frame rate (used to convert per-frame delta to per-second).
        """
        if (
            state.bbox is None
            or state.meter_position is None
            or state.prev_meter_pos is None
        ):
            return

        # Compute instantaneous speed from frame-to-frame displacement
        delta_meters = euclidean_dist(state.meter_position, state.prev_meter_pos)
        speed_mps = delta_meters * max(fps, 1.0)
        speed_kmh = speed_mps * 3.6

        x1, y1, x2, y2 = state.bbox
        x1, y2 = int(x1), int(y2)

        # Position below the bounding box
        label = f"~{speed_kmh:.0f} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        label_x = x1
        label_y = y2 + 18  # Below the bbox bottom

        # Ensure label doesn't go below the frame
        frame_h = frame.shape[0]
        if label_y > frame_h - 5:
            label_y = int(state.bbox[1]) - 25  # Place above bbox instead

        # Shadow for contrast
        cv2.putText(
            frame, label, (label_x + 1, label_y + 1),
            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        # White foreground text
        cv2.putText(
            frame, label, (label_x, label_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    def draw_player(
        self,
        frame: np.ndarray,
        state: PlayerState,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw bounding box, label, and foot-point dot for one player.

        The label uses a shadow technique: first draw in black offset by
        (1, 1) pixels, then draw the colored text on top. This ensures
        readability regardless of background brightness.

        Args:
            frame: The video frame to draw on (mutated in-place).
            state: Current PlayerState for this player.
            color: BGR color tuple for this player's overlays.
            label: Text to display above the bbox (e.g. "P1: 47.23 m").
        """
        if state.bbox is None:
            # Player has never been detected — nothing to draw
            return

        x1, y1, x2, y2 = state.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # --- BOUNDING BOX ---
        # Thickness=2 provides visibility without obscuring the player.
        # LINE_AA (anti-aliased) prevents jagged edges on angled boxes.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

        # --- PLAYER LABEL ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Position the label above the bounding box with a small margin
        label_x = x1
        label_y = y1 - 10  # 10px above the top of the bbox

        # Ensure label doesn't go above the frame
        if label_y < 20:
            label_y = y2 + 25  # Place below bbox instead

        # Shadow text (black, offset by 1px) for contrast on any background
        cv2.putText(
            frame, label, (label_x + 1, label_y + 1),
            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        # Foreground text in the player's color
        cv2.putText(
            frame, label, (label_x, label_y),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )

        # --- FOOT POINT DOT ---
        # Small filled circle at the bottom-center of the bbox to visually
        # confirm that the tracking anchor is at the player's feet.
        if state.smoothed_foot is not None:
            foot_pt = (int(state.smoothed_foot[0]), int(state.smoothed_foot[1]))
            cv2.circle(frame, foot_pt, 4, color, -1, lineType=cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        p1: PlayerState,
        p2: PlayerState,
        frame_idx: int,
        total_frames: int,
        fps: float,
        video_name: str,
    ) -> None:
        """Draw the heads-up display panel in the top-left corner.

        The HUD shows video metadata and live distance counters for both
        players. It uses a semi-transparent dark background so text is
        readable over any court surface or broadcast graphic.

        Args:
            frame: The video frame to draw on (mutated in-place).
            p1: Player 1 state.
            p2: Player 2 state.
            frame_idx: Current frame index (0-based).
            total_frames: Total number of frames in the video.
            fps: Video frame rate.
            video_name: Display name of the video (stem without extension).
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 28  # vertical spacing between text lines

        # Compute elapsed and total duration from frame index and FPS
        elapsed_sec = frame_idx / max(fps, 1.0)
        total_sec = total_frames / max(fps, 1.0)

        # Build the HUD text lines
        lines = [
            f"Video: {video_name}",
            f"Frame: {frame_idx + 1} / {total_frames}",
            f"Time:  {format_time(elapsed_sec)} / {format_time(total_sec)}",
            # Unicode box-drawing character for a clean separator
            "\u2501" * 22,
            f"P1 Distance: {p1.total_distance_m:.2f} m",
            f"P2 Distance: {p2.total_distance_m:.2f} m",
        ]

        # Calculate the panel dimensions based on the widest text line
        max_text_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_text_width = max(max_text_width, tw)

        # Panel padding and margins
        pad_x = 15     # horizontal padding inside the panel
        pad_y = 12     # vertical padding above first line and below last
        margin = 10    # offset of the panel from the frame edge

        panel_w = max_text_width + 2 * pad_x
        panel_h = len(lines) * line_height + 2 * pad_y

        # --- Draw the semi-transparent background rectangle ---
        # Create an overlay for alpha blending (same technique as trail drawing)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + panel_w, margin + panel_h),
            (0, 0, 0),   # Black background
            -1,           # Filled rectangle
        )
        # Blend with the original frame at the configured opacity
        cv2.addWeighted(
            overlay, config.HUD_ALPHA,
            frame, 1.0 - config.HUD_ALPHA,
            0,
            frame,
        )

        # --- Draw text lines on top of the blended background ---
        for i, line in enumerate(lines):
            text_x = margin + pad_x
            text_y = margin + pad_y + (i + 1) * line_height

            cv2.putText(
                frame, line, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

    def draw_stats_panel(
        self,
        frame: np.ndarray,
        p1: PlayerState,
        p2: PlayerState,
        fps: float,
    ) -> None:
        """Draw a player statistics panel showing speed and distance.

        Positioned in the bottom-left corner with a semi-transparent background.
        Shows current speed, average speed, and total distance for both players.
        Inspired by the tennis_analysis reference repo's draw_player_stats.

        Args:
            frame: The video frame to draw on (mutated in-place).
            p1: Player 1 state.
            p2: Player 2 state.
            fps: Video FPS for speed calculation.
        """
        frame_h, frame_w = frame.shape[:2]

        # Panel dimensions
        panel_w = 320
        panel_h = 100
        margin = 15

        x1 = margin
        y1 = frame_h - margin - panel_h
        x2 = x1 + panel_w
        y2 = frame_h - margin

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(frame, "         Player 1    Player 2",
                     (x1 + 10, y1 + 20), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Speed row
        def get_speed(state: PlayerState) -> float:
            if (state.prev_meter_pos is not None
                    and state.meter_position is not None
                    and fps > 0):
                spd = euclidean_dist(state.meter_position, state.prev_meter_pos) * fps * 3.6
                return min(spd, 40.0)  # cap display at 40 km/h
            return 0.0

        p1_spd = get_speed(p1)
        p2_spd = get_speed(p2)
        cv2.putText(frame, "Speed", (x1 + 10, y1 + 45), font, 0.4,
                     (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{p1_spd:5.1f} km/h   {p2_spd:5.1f} km/h",
                     (x1 + 90, y1 + 45), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Distance row
        cv2.putText(frame, "Distance", (x1 + 10, y1 + 70), font, 0.4,
                     (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{p1.total_distance_m:5.1f} m     {p2.total_distance_m:5.1f} m",
                     (x1 + 90, y1 + 70), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Missed frames row
        cv2.putText(frame, "Status", (x1 + 10, y1 + 90), font, 0.35,
                     (200, 200, 200), 1, cv2.LINE_AA)
        p1_status = "OK" if p1.missed_frames == 0 else f"LOST({p1.missed_frames}f)"
        p2_status = "OK" if p2.missed_frames == 0 else f"LOST({p2.missed_frames}f)"
        cv2.putText(frame, f"{p1_status:>10s}   {p2_status:>10s}",
                     (x1 + 90, y1 + 90), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        p1: PlayerState,
        p2: PlayerState,
        frame_idx: int,
        total_frames: int,
        fps: float,
        video_name: str,
    ) -> np.ndarray:
        """Render all overlays onto a frame and return the annotated result.

        Draw order is carefully chosen so background elements (trails) don't
        obscure foreground elements (text, boxes):
          1. Trails (background)
          2. Missed-frame indicators (if applicable — dashed borders)
          3. Player overlays (mid-ground — bbox, label, foot dot)
          4. Score bars (above each player's bbox)
          5. Speed indicators (below each player's bbox)
          6. HUD panel (foreground, top-left)
          7. Mini-court (bottom-right)
          8. Stats panel (bottom-left)

        Args:
            frame: Original BGR frame from the video.
            p1: Player 1 state.
            p2: Player 2 state.
            frame_idx: Current frame index.
            total_frames: Total frames in the video.
            fps: Video FPS.
            video_name: Video stem name for display.

        Returns:
            The annotated frame (same object as input, mutated in-place).
        """
        # 1. TRAJECTORY TRAILS — drawn first so they appear behind everything
        self.draw_trail(frame, p1.trail, config.P1_COLOR)
        self.draw_trail(frame, p2.trail, config.P2_COLOR)

        # 2. MISSED-FRAME INDICATORS — dashed border when player is temporarily lost
        self.draw_missed_frame_indicator(frame, p1)
        self.draw_missed_frame_indicator(frame, p2)

        # 3. PLAYER OVERLAYS — bounding box, label, foot dot
        p1_label = f"P1: {p1.total_distance_m:.2f} m"
        p2_label = f"P2: {p2.total_distance_m:.2f} m"

        self.draw_player(frame, p1, config.P1_COLOR, p1_label)
        self.draw_player(frame, p2, config.P2_COLOR, p2_label)

        # 4. SCORE BARS — thin colored bar above bbox showing detection quality
        self.draw_score_bar(frame, p1)
        self.draw_score_bar(frame, p2)

        # 5. SPEED INDICATORS — estimated km/h below each player
        self.draw_speed_indicator(frame, p1, config.P1_COLOR, fps)
        self.draw_speed_indicator(frame, p2, config.P2_COLOR, fps)

        # 6. HUD PANEL — always on top
        self.draw_hud(frame, p1, p2, frame_idx, total_frames, fps, video_name)

        # 7. MINI-COURT — bird's-eye view in bottom-right
        self._mini_court.draw(frame, p1.meter_position, p2.meter_position)

        # 8. STATS PANEL — bottom-left
        self.draw_stats_panel(frame, p1, p2, fps)

        return frame
