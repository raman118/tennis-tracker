"""
Mini-court bird's-eye view visualization module.

Draws a scaled-down tennis court diagram in the corner of the video frame
and plots player positions as colored dots, providing an intuitive overhead
view of player movement. Inspired by the tennis_analysis reference repo's
MiniCourt class but simplified to use our existing homography-based
meter-space coordinates.

Architecture decision: the mini-court maps meter-space positions directly
to pixel coordinates on a pre-drawn court diagram. This is simpler and more
accurate than the reference repo's approach of using player height as a
reference scale, since we already have a homography that gives us meters.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from tennis_tracker import config

logger = logging.getLogger(__name__)


class MiniCourt:
    """Draws a bird's-eye mini-court overlay with player position dots.

    The mini-court is positioned in the bottom-right corner of the frame.
    It shows a proportionally-scaled singles court with service boxes,
    baselines, and net, plus colored dots for each player's current position.
    """

    # Drawing constants
    COURT_WIDTH_PX: int = 120     # width of the mini-court in pixels
    COURT_HEIGHT_PX: int = 260    # height of the mini-court in pixels
    PADDING: int = 15             # padding inside the background box
    MARGIN: int = 15              # margin from frame edge
    BG_ALPHA: float = 0.6        # background transparency

    def __init__(self) -> None:
        """Initialize the mini-court renderer.

        Pre-computes the scaling factors from real-world meters to
        mini-court pixels based on ITF standard court dimensions.
        """
        # Scale factors: meters → mini-court pixels
        self._scale_x = self.COURT_WIDTH_PX / config.COURT_WIDTH_M
        self._scale_y = self.COURT_HEIGHT_PX / config.COURT_LENGTH_M

        logger.info("MiniCourt initialized (%.0f x %.0f px).",
                     self.COURT_WIDTH_PX, self.COURT_HEIGHT_PX)

    def meter_to_mini_court(
        self,
        meter_pos: Tuple[float, float],
        court_origin_x: int,
        court_origin_y: int,
    ) -> Tuple[int, int]:
        """Convert a real-world meter position to mini-court pixel coordinates.

        The meter-space origin (0, 0) corresponds to the top-left corner of
        the court (near baseline left sideline). X increases rightward along
        the baseline, Y increases downward toward the far baseline.

        Args:
            meter_pos: (mx, my) position in meters from the homography.
            court_origin_x: Pixel X of the mini-court's top-left corner.
            court_origin_y: Pixel Y of the mini-court's top-left corner.

        Returns:
            (px, py) pixel coordinates on the mini-court.
        """
        mx, my = meter_pos
        # Clamp to court boundaries to prevent dots from going outside
        mx = max(0.0, min(mx, config.COURT_WIDTH_M))
        my = max(0.0, min(my, config.COURT_LENGTH_M))

        px = court_origin_x + int(mx * self._scale_x)
        py = court_origin_y + int(my * self._scale_y)
        return (px, py)

    def draw(
        self,
        frame: np.ndarray,
        p1_meter_pos: Optional[Tuple[float, float]],
        p2_meter_pos: Optional[Tuple[float, float]],
    ) -> None:
        """Draw the mini-court overlay with player positions onto the frame.

        Draws a semi-transparent background, a scaled court diagram with
        baselines, sidelines, service boxes, center line, and net, then
        plots colored dots for each player.

        Args:
            frame: The video frame to draw on (mutated in-place).
            p1_meter_pos: Player 1 position in meters, or None if unknown.
            p2_meter_pos: Player 2 position in meters, or None if unknown.
        """
        frame_h, frame_w = frame.shape[:2]

        # Compute the background box position (bottom-right corner)
        box_w = self.COURT_WIDTH_PX + 2 * self.PADDING
        box_h = self.COURT_HEIGHT_PX + 2 * self.PADDING + 20  # +20 for title

        box_x2 = frame_w - self.MARGIN
        box_y2 = frame_h - self.MARGIN
        box_x1 = box_x2 - box_w
        box_y1 = box_y2 - box_h

        # Court origin (top-left corner of the actual court drawing)
        court_x = box_x1 + self.PADDING
        court_y = box_y1 + self.PADDING + 18  # +18 for title text

        # --- Semi-transparent background ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2),
                       (30, 30, 30), -1)
        cv2.addWeighted(overlay, self.BG_ALPHA, frame,
                         1.0 - self.BG_ALPHA, 0, frame)

        # --- Title ---
        cv2.putText(frame, "Court View", (box_x1 + 10, box_y1 + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                     cv2.LINE_AA)

        # --- Draw court outline (baselines + sidelines) ---
        court_color = (0, 180, 0)  # Green court lines
        court_fill = (34, 100, 34)  # Dark green court surface

        # Fill court surface
        cv2.rectangle(frame,
                       (court_x, court_y),
                       (court_x + self.COURT_WIDTH_PX, court_y + self.COURT_HEIGHT_PX),
                       court_fill, -1)

        # Court outline (baselines + sidelines)
        cv2.rectangle(frame,
                       (court_x, court_y),
                       (court_x + self.COURT_WIDTH_PX, court_y + self.COURT_HEIGHT_PX),
                       (255, 255, 255), 1, cv2.LINE_AA)

        # --- Net (horizontal line at center) ---
        net_y = court_y + self.COURT_HEIGHT_PX // 2
        cv2.line(frame,
                  (court_x, net_y),
                  (court_x + self.COURT_WIDTH_PX, net_y),
                  (200, 200, 200), 2, cv2.LINE_AA)

        # --- Service boxes ---
        # Service line height: 6.4m from each baseline = 5.485m from net
        service_line_frac = 6.40 / config.COURT_LENGTH_M
        service_top_y = court_y + int(service_line_frac * self.COURT_HEIGHT_PX)
        service_bot_y = court_y + self.COURT_HEIGHT_PX - int(service_line_frac * self.COURT_HEIGHT_PX)

        # Top service line
        cv2.line(frame,
                  (court_x, service_top_y),
                  (court_x + self.COURT_WIDTH_PX, service_top_y),
                  (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom service line
        cv2.line(frame,
                  (court_x, service_bot_y),
                  (court_x + self.COURT_WIDTH_PX, service_bot_y),
                  (255, 255, 255), 1, cv2.LINE_AA)

        # Center service line (vertical, between service lines)
        center_x = court_x + self.COURT_WIDTH_PX // 2
        cv2.line(frame,
                  (center_x, service_top_y),
                  (center_x, service_bot_y),
                  (255, 255, 255), 1, cv2.LINE_AA)

        # Center mark on baselines (short marks)
        mark_len = 6
        cv2.line(frame, (center_x, court_y), (center_x, court_y + mark_len),
                  (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (center_x, court_y + self.COURT_HEIGHT_PX),
                  (center_x, court_y + self.COURT_HEIGHT_PX - mark_len),
                  (255, 255, 255), 1, cv2.LINE_AA)

        # --- Player dots ---
        dot_radius = 5

        if p1_meter_pos is not None:
            p1_px = self.meter_to_mini_court(p1_meter_pos, court_x, court_y)
            cv2.circle(frame, p1_px, dot_radius, config.P1_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, p1_px, dot_radius, (255, 255, 255), 1, cv2.LINE_AA)

        if p2_meter_pos is not None:
            p2_px = self.meter_to_mini_court(p2_meter_pos, court_x, court_y)
            cv2.circle(frame, p2_px, dot_radius, config.P2_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, p2_px, dot_radius, (255, 255, 255), 1, cv2.LINE_AA)
