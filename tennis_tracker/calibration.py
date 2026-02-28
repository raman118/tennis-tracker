"""
Court calibration module for the Tennis Player Tracking pipeline.

Maps pixel coordinates to real-world meters using a homography transformation
derived from the four corners of a standard ITF singles tennis court.

The calibration cascade is:
  1. AUTO-DETECT — Canny edge detection + Hough line intersection
  2. MANUAL CLICK — OpenCV window where the user clicks 4 court corners
  3. PIXEL FALLBACK — No transformation; distances remain in pixel units

Architecture decision: the cascade approach ensures the pipeline always
produces output. Auto-detection works well on clean broadcast footage with
sharp white court lines, but broadcast graphics or non-standard cameras
may require manual input or graceful degradation.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tennis_tracker import config

logger = logging.getLogger(__name__)


class CourtCalibrator:
    """Establishes the pixel-to-meter mapping for a tennis court.

    Given the first frame of a video, this class attempts to locate the four
    corners of the singles court and compute a homography matrix that maps
    any pixel coordinate to real-world meters.

    Attributes:
        H: 3×3 homography matrix (None if calibration failed entirely).
        calibration_mode: One of 'homography_auto', 'homography_manual',
                          or 'pixel_fallback'.
        pixel_corners: The four court corners in pixel space (for court mask).
    """

    # Real-world meter coordinates of the four singles court corners.
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    # This matches the visual layout when the camera looks down the court
    # from an elevated broadcast position.
    METER_CORNERS: np.ndarray = np.array(
        [
            [0.0, 0.0],                                # Top-Left
            [config.COURT_WIDTH_M, 0.0],               # Top-Right
            [config.COURT_WIDTH_M, config.COURT_LENGTH_M],  # Bottom-Right
            [0.0, config.COURT_LENGTH_M],              # Bottom-Left
        ],
        dtype=np.float32,
    )

    def __init__(self, frame: np.ndarray, interactive: bool = True) -> None:
        """Initialize the calibrator using the first video frame.

        Args:
            frame: First frame of the video (BGR, uint8).
            interactive: If True, allow manual corner clicking as fallback.
                         Set to False for headless / batch processing.
        """
        self.H: Optional[np.ndarray] = None
        self.calibration_mode: str = "pixel_fallback"
        self.pixel_corners: Optional[np.ndarray] = None
        self._expanded_contour: Optional[np.ndarray] = None  # cached court polygon

        # Step 1: Try automatic corner detection
        corners = self._auto_detect_corners(frame)
        manual = False

        # Step 2: If auto failed and we're allowed to ask the user, open a GUI
        if corners is None and interactive:
            logger.info("Auto-detection failed — switching to manual corner selection.")
            corners = self._manual_click_corners(frame)
            manual = True

        # Step 3: Compute homography if we have corners, else degrade gracefully
        if corners is not None:
            self.pixel_corners = np.array(corners, dtype=np.float32)
            self._compute_homography(corners)
            if manual:
                self.set_mode_manual()
        else:
            logger.warning(
                "Court calibration failed entirely — using pixel fallback mode. "
                "Distances will be in pixel units, not meters."
            )

    # ------------------------------------------------------------------
    # Auto-detect via Canny + Hough
    # ------------------------------------------------------------------

    def _auto_detect_corners(
        self, frame: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """Detect the four singles court corners automatically.

        Pipeline:
          Grayscale → GaussianBlur → Canny → HoughLinesP →
          Filter by angle (horizontal / vertical) →
          Compute line intersections → Pick 4 outermost corners →
          Sanity-check aspect ratio.

        Returns:
            List of 4 (x, y) tuples in TL, TR, BR, BL order, or None on failure.
        """
        # Convert to grayscale — edge detection operates on intensity only
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian blur suppresses high-frequency noise (compression artifacts,
        # texture on the court surface) so Canny doesn't produce spurious edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny thresholds (50, 150) are a widely-used default for broadcast
        # video. The 1:3 ratio follows Canny's recommended hysteresis range.
        edges = cv2.Canny(blurred, 50, 150)

        # HoughLinesP returns line segments. Parameters tuned for court lines:
        # - rho=1: 1-pixel resolution in the accumulator
        # - theta=pi/180: 1-degree angular resolution
        # - threshold=80: minimum votes (lower catches faint lines)
        # - minLineLength=100: court lines are long; rejects short noise
        # - maxLineGap=30: bridges small gaps in dashed service lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=100,
            maxLineGap=30,
        )

        if lines is None or len(lines) < 4:
            logger.warning("HoughLinesP found fewer than 4 lines — auto-detect failed.")
            return None

        # Classify lines by angle into horizontal and vertical groups.
        # Tennis court lines viewed from a broadcast angle are approximately
        # horizontal (baselines) and near-vertical (sidelines), though
        # perspective makes verticals converge slightly.
        horizontal_lines: List[np.ndarray] = []
        vertical_lines: List[np.ndarray] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Compute angle in degrees from horizontal
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Near-horizontal: within 30° of the x-axis
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])
            # Near-vertical: within 30° of the y-axis (i.e., 60°-120°)
            elif 60 < angle < 120:
                vertical_lines.append(line[0])
            # Lines at intermediate angles (diagonals) are ignored — they are
            # likely service lines or perspective-skewed non-boundary lines

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.warning(
                "Insufficient horizontal (%d) or vertical (%d) lines found.",
                len(horizontal_lines),
                len(vertical_lines),
            )
            return None

        # Compute intersections between every horizontal-vertical pair.
        # Each intersection is a candidate court corner.
        intersections = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                pt = self._line_intersection(h_line, v_line)
                if pt is not None:
                    px, py = pt
                    # Filter intersections that fall outside the frame
                    h, w = frame.shape[:2]
                    if 0 <= px <= w and 0 <= py <= h:
                        intersections.append(pt)

        if len(intersections) < 4:
            logger.warning(
                "Only %d valid intersections found (need ≥ 4) — auto-detect failed.",
                len(intersections),
            )
            return None

        # Identify the 4 outermost corner candidates.
        # Strategy: compute the centroid of all intersections, then find the
        # single point that is farthest in each quadrant (TL, TR, BR, BL).
        corners = self._select_four_corners(intersections, frame.shape)

        if corners is None:
            return None

        # Sanity check: the detected quadrilateral should have an aspect ratio
        # close to the known court ratio of length/width = 23.77/8.23 ≈ 2.89.
        # We allow a generous tolerance (1.5 – 5.0) because perspective
        # distortion significantly warps the apparent ratio.
        if not self._sanity_check(corners):
            logger.warning("Court corner aspect ratio sanity check failed.")
            return None

        logger.info("Auto-detected 4 court corners successfully.")
        return corners

    @staticmethod
    def _line_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Find the intersection point of two line segments.

        Uses the parametric form of line-line intersection. Returns None if
        lines are parallel (determinant near zero).

        Args:
            line1: (x1, y1, x2, y2) endpoints of line segment 1.
            line2: (x1, y1, x2, y2) endpoints of line segment 2.

        Returns:
            (x, y) intersection point, or None if lines are parallel.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Direction vectors of each line
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3

        # Cross product of the two direction vectors. Zero means parallel.
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-6:
            return None

        # Solve for the parameter t along line1 where intersection occurs
        t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom

        # Compute intersection coordinates using parametric line equation
        ix = x1 + t * dx1
        iy = y1 + t * dy1
        return (float(ix), float(iy))

    @staticmethod
    def _select_four_corners(
        intersections: List[Tuple[float, float]],
        frame_shape: tuple,
    ) -> Optional[List[Tuple[float, float]]]:
        """Select the four outermost intersection points as court corners.

        Partitions points into quadrants relative to the frame center, then
        picks the point in each quadrant that is farthest from center. This
        identifies the bounding quadrilateral of the court.

        Args:
            intersections: All detected line-intersection points.
            frame_shape: (height, width, channels) of the frame.

        Returns:
            [TL, TR, BR, BL] corner tuples, or None if any quadrant is empty.
        """
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Partition intersections into four screen quadrants
        tl_pts = [(x, y) for x, y in intersections if x <= cx and y <= cy]
        tr_pts = [(x, y) for x, y in intersections if x > cx and y <= cy]
        br_pts = [(x, y) for x, y in intersections if x > cx and y > cy]
        bl_pts = [(x, y) for x, y in intersections if x <= cx and y > cy]

        # Each quadrant must have at least one point to form a court rectangle
        if not all([tl_pts, tr_pts, br_pts, bl_pts]):
            return None

        # Select the corner in each quadrant farthest from center — this
        # gives us the outermost (most extreme) court boundary point
        def farthest(pts: list) -> Tuple[float, float]:
            return max(pts, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

        return [farthest(tl_pts), farthest(tr_pts), farthest(br_pts), farthest(bl_pts)]

    def _sanity_check(self, corners: List[Tuple[float, float]]) -> bool:
        """Verify that detected corners form a plausible tennis court shape.

        Computes a rough aspect ratio of the bounding quadrilateral and
        checks it against the expected ~2.89 ratio with generous tolerance.

        Args:
            corners: [TL, TR, BR, BL] pixel coordinates.

        Returns:
            True if the aspect ratio is within acceptable bounds.
        """
        tl, tr, br, bl = corners

        # Average width: mean of top edge and bottom edge lengths
        top_width = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        bot_width = np.hypot(br[0] - bl[0], br[1] - bl[1])
        avg_width = (top_width + bot_width) / 2.0

        # Average height: mean of left edge and right edge lengths
        left_height = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
        right_height = np.hypot(br[0] - tr[0], br[1] - tr[1])
        avg_height = (left_height + right_height) / 2.0

        if avg_width < 1e-6:
            return False

        # The true ratio is 23.77/8.23 ≈ 2.89, but due to perspective
        # distortion from the broadcast camera angle, the apparent ratio
        # can deviate substantially. We use a wide tolerance of 0.5-8.0
        # to accommodate steep viewing angles while still rejecting
        # clearly wrong detections (e.g., detecting the doubles court
        # or stadium structures).
        aspect = avg_height / avg_width
        is_valid = 0.5 < aspect < 8.0
        logger.debug("Detected court aspect ratio: %.2f (valid=%s)", aspect, is_valid)
        return is_valid

    # ------------------------------------------------------------------
    # Manual click fallback
    # ------------------------------------------------------------------

    def _manual_click_corners(
        self, frame: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """Open an interactive window for the user to click 4 court corners.

        Displays the first frame with instructions and collects exactly 4
        mouse clicks. Each click is visualized as a colored circle with a
        label (TL, TR, BR, BL). Press 'r' to reset all clicks.

        Args:
            frame: The first video frame to display.

        Returns:
            List of 4 (x, y) tuples, or None if the user closed the window.
        """
        # Clone the frame so we can draw on it without mutating the original
        display = frame.copy()
        corners: List[Tuple[float, float]] = []

        # Labels and colors for each corner in click order
        corner_labels = ["TL", "TR", "BR", "BL"]
        corner_colors = [
            (0, 255, 0),    # Green — top-left
            (255, 0, 0),    # Blue — top-right
            (0, 0, 255),    # Red — bottom-right
            (255, 255, 0),  # Cyan — bottom-left
        ]

        window_name = "Court Calibration — Click 4 corners: TL, TR, BR, BL | 'r' to reset"

        def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
            """Handles mouse click events to collect corner coordinates."""
            nonlocal display
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((float(x), float(y)))
                idx = len(corners) - 1

                # Draw a filled circle at the click position for visual feedback
                cv2.circle(display, (x, y), 8, corner_colors[idx], -1)

                # Draw a label next to the circle so the user knows which corner it is
                cv2.putText(
                    display,
                    corner_labels[idx],
                    (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    corner_colors[idx],
                    2,
                )
                cv2.imshow(window_name, display)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display)
        cv2.setMouseCallback(window_name, mouse_callback)

        # Draw instruction text on the frame
        cv2.putText(
            display,
            "Click 4 court corners: TL, TR, BR, BL. Press 'r' to reset.",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, display)

        while True:
            key = cv2.waitKey(50) & 0xFF

            # 'r' key resets all collected corners
            if key == ord("r"):
                corners.clear()
                display = frame.copy()
                cv2.putText(
                    display,
                    "Click 4 court corners: TL, TR, BR, BL. Press 'r' to reset.",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window_name, display)
                logger.info("Corner selection reset by user.")

            # Check if all 4 corners have been collected
            if len(corners) == 4:
                break

            # Check if the window was closed (e.g., user hit the X button)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.warning("Calibration window closed by user before 4 corners were selected.")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        logger.info(
            "Court calibration: MANUAL mode — user provided 4 corners: %s", corners
        )
        return corners

    # ------------------------------------------------------------------
    # Homography computation
    # ------------------------------------------------------------------

    def _compute_homography(self, corners: List[Tuple[float, float]]) -> None:
        """Compute the 3×3 homography matrix from pixel corners to meter corners.

        Uses cv2.findHomography which internally solves the Direct Linear
        Transform (DLT) to find the projective transformation that maps the
        four pixel-space corners to the known meter-space court rectangle.

        Args:
            corners: [TL, TR, BR, BL] pixel coordinates.
        """
        pixel_pts = np.array(corners, dtype=np.float32)
        meter_pts = self.METER_CORNERS

        # findHomography returns (H, status_mask). We ignore the mask because
        # we have exactly 4 point correspondences (the minimum for a unique
        # homography) so there are no outliers to mask out.
        H, _ = cv2.findHomography(pixel_pts, meter_pts)

        if H is not None:
            self.H = H
            # Determine mode based on how corners were obtained
            self.calibration_mode = "homography_auto"
            logger.info(
                "Homography computed successfully (mode: %s).", self.calibration_mode
            )
        else:
            logger.warning("cv2.findHomography returned None — falling back to pixel mode.")
            self.calibration_mode = "pixel_fallback"

    def set_mode_manual(self) -> None:
        """Override the calibration mode label to 'homography_manual'.

        Called when corners were obtained via manual click rather than
        auto-detection, after the homography has been computed.
        """
        if self.H is not None:
            self.calibration_mode = "homography_manual"

    # ------------------------------------------------------------------
    # Runtime transforms
    # ------------------------------------------------------------------

    def transform(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a pixel coordinate to real-world meters.

        Delegates to utils.pixel_to_meter but is provided as a method on the
        calibrator for a cleaner API (callers don't need to pass H around).

        Args:
            pixel_point: (px, py) position in pixel space.

        Returns:
            (mx, my) in meters, or raw (px, py) in fallback mode.
        """
        from tennis_tracker.utils import pixel_to_meter
        return pixel_to_meter(pixel_point, self.H)

    def is_inside_court(self, pixel_point: Tuple[float, float]) -> bool:
        """Check whether a pixel point lies inside an EXPANDED court polygon.

        The expansion is critical: auto-detected corners lie on the court
        lines, but a near-baseline player's foot point (bottom-center of bbox)
        often falls a few pixels below the detected baseline due to
        perspective distortion and bbox imprecision. Without expansion, the
        near-baseline player is falsely classified as outside the court and
        gets filtered out — which cascades into identity contamination when
        non-players fill the empty slot via nearest-centroid recovery.

        cv2.pointPolygonTest returns:
          > 0  if inside the polygon
          = 0  if on the boundary
          < 0  if outside the polygon

        Args:
            pixel_point: (px, py) to test.

        Returns:
            True if the point is inside the expanded court polygon.
        """
        if self.pixel_corners is None:
            # If no court polygon is available, assume all points are valid.
            # This prevents the pipeline from discarding all detections.
            return True

        # Compute and cache the expanded contour on first call
        if self._expanded_contour is None:
            self._expanded_contour = self._compute_expanded_contour()

        # measureDist=False because we only need inside/outside, not the distance
        result = cv2.pointPolygonTest(
            self._expanded_contour, pixel_point, measureDist=False
        )

        # result >= 0 means the point is inside or on the boundary
        return result >= 0

    def _compute_expanded_contour(self) -> np.ndarray:
        """Expand the court polygon outward from its centroid by a percentage margin.

        Each corner is pushed further from the centroid by COURT_MARGIN_PERCENT.
        This creates a larger inclusion zone that accounts for bbox imprecision
        at the court boundaries.

        Returns:
            Expanded contour as an int32 array suitable for cv2.pointPolygonTest.
        """
        corners = self.pixel_corners  # shape (4, 2)

        # Compute centroid of the court polygon
        centroid_x = float(np.mean(corners[:, 0]))
        centroid_y = float(np.mean(corners[:, 1]))

        margin = config.COURT_MARGIN_PERCENT
        expanded = []

        for px, py in corners:
            # Vector from centroid to corner
            dx = float(px) - centroid_x
            dy = float(py) - centroid_y

            # Push corner outward by margin percentage
            new_x = centroid_x + dx * (1.0 + margin)
            new_y = centroid_y + dy * (1.0 + margin)
            expanded.append([new_x, new_y])

        expanded_arr = np.array(expanded, dtype=np.float32)
        contour = expanded_arr.reshape(-1, 1, 2).astype(np.int32)

        logger.info(
            "Court polygon expanded by %.0f%% for player inclusion checks. "
            "Original corners: %s, Expanded: %s",
            margin * 100,
            corners.tolist(),
            expanded_arr.tolist(),
        )

        return contour
