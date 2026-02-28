"""
Court heatmap image generator for the Tennis Player Tracking pipeline.

Produces publication-quality heatmap images showing player court coverage
density, overlaid on a proportionally accurate court diagram. Each player
gets their own heatmap image saved alongside the JSON report.

The heatmap uses a perceptually uniform colormap (blue → yellow → red)
to represent position density, with the actual tennis court lines drawn
underneath for spatial reference.

Architecture decision: heatmap generation is decoupled from the analytics
engine so it can be run independently or skipped in headless environments
where matplotlib is not available.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from tennis_tracker import config

logger = logging.getLogger(__name__)

# Court line positions in meters (ITF standard singles court)
_COURT_W = config.COURT_WIDTH_M   # 8.23
_COURT_L = config.COURT_LENGTH_M  # 23.77
_SERVICE_LINE = 6.40               # Distance from baseline to service line
_CENTER_MARK = 0.10                # Small tick mark at center of baseline


def generate_heatmap_image(
    heatmap_data: Optional[np.ndarray],
    player_id: int,
    output_path: str,
    img_width: int = 400,
    img_height: int = 900,
) -> Optional[str]:
    """Generate and save a heatmap image overlaid on a court diagram.

    Uses pure OpenCV (no matplotlib dependency) to create a professional-
    looking heatmap visualization with Gaussian blur for smooth density
    rendering and court lines drawn underneath.

    Args:
        heatmap_data: 2D numpy array from AnalyticsEngine (bins_x × bins_y).
        player_id: 1 or 2, used for title and color scheme.
        output_path: Full path where the PNG image will be saved.
        img_width: Output image width in pixels.
        img_height: Output image height in pixels.

    Returns:
        The output path string if successful, None otherwise.
    """
    if heatmap_data is None or heatmap_data.max() == 0:
        logger.warning("No heatmap data for Player %d — skipping image generation.", player_id)
        return None

    # --- Create base court image ---
    court_img = _draw_court_background(img_width, img_height)

    # --- Generate heatmap overlay ---
    heatmap_overlay = _render_heatmap_overlay(heatmap_data, img_width, img_height, player_id)

    # --- Composite heatmap over court ---
    # Use addWeighted so court lines show through
    result = cv2.addWeighted(court_img, 0.4, heatmap_overlay, 0.6, 0)

    # --- Redraw court lines on top for clarity ---
    _draw_court_lines(result, img_width, img_height, line_alpha=0.8)

    # --- Add title and legend ---
    _draw_title_and_legend(result, player_id, img_width, heatmap_data.max())

    # --- Save ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    logger.info("Heatmap saved for Player %d: %s", player_id, output_path)

    return output_path


def _draw_court_background(w: int, h: int) -> np.ndarray:
    """Create a dark court background image."""
    img = np.full((h, w, 3), (40, 45, 40), dtype=np.uint8)  # Dark green-gray

    # Court surface (slightly lighter green)
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.05)
    cv2.rectangle(
        img,
        (margin_x, margin_y),
        (w - margin_x, h - margin_y),
        (45, 80, 45),
        -1,
    )
    return img


def _meter_to_pixel(
    mx: float, my: float, img_w: int, img_h: int
) -> Tuple[int, int]:
    """Convert meter coordinates to pixel coordinates on the heatmap image."""
    margin_x = int(img_w * 0.08)
    margin_y = int(img_h * 0.05)
    court_w_px = img_w - 2 * margin_x
    court_h_px = img_h - 2 * margin_y

    px = margin_x + int(mx / _COURT_W * court_w_px)
    py = margin_y + int(my / _COURT_L * court_h_px)
    return (px, py)


def _draw_court_lines(
    img: np.ndarray, w: int, h: int, line_alpha: float = 1.0
) -> None:
    """Draw ITF-accurate court lines onto the image."""
    white = (255, 255, 255)
    thickness = 2

    if line_alpha < 1.0:
        overlay = img.copy()
        _draw_court_lines_impl(overlay, w, h, white, thickness)
        cv2.addWeighted(overlay, line_alpha, img, 1.0 - line_alpha, 0, img)
    else:
        _draw_court_lines_impl(img, w, h, white, thickness)


def _draw_court_lines_impl(
    img: np.ndarray, w: int, h: int,
    color: Tuple[int, int, int], thickness: int
) -> None:
    """Internal: draw all court line segments."""
    # Baseline top
    tl = _meter_to_pixel(0, 0, w, h)
    tr = _meter_to_pixel(_COURT_W, 0, w, h)
    cv2.line(img, tl, tr, color, thickness, cv2.LINE_AA)

    # Baseline bottom
    bl = _meter_to_pixel(0, _COURT_L, w, h)
    br = _meter_to_pixel(_COURT_W, _COURT_L, w, h)
    cv2.line(img, bl, br, color, thickness, cv2.LINE_AA)

    # Left sideline
    cv2.line(img, tl, bl, color, thickness, cv2.LINE_AA)

    # Right sideline
    cv2.line(img, tr, br, color, thickness, cv2.LINE_AA)

    # Net
    net_l = _meter_to_pixel(0, _COURT_L / 2.0, w, h)
    net_r = _meter_to_pixel(_COURT_W, _COURT_L / 2.0, w, h)
    cv2.line(img, net_l, net_r, (200, 200, 200), thickness + 1, cv2.LINE_AA)

    # Service line top
    sl_top_l = _meter_to_pixel(0, _SERVICE_LINE, w, h)
    sl_top_r = _meter_to_pixel(_COURT_W, _SERVICE_LINE, w, h)
    cv2.line(img, sl_top_l, sl_top_r, color, thickness, cv2.LINE_AA)

    # Service line bottom
    sl_bot_l = _meter_to_pixel(0, _COURT_L - _SERVICE_LINE, w, h)
    sl_bot_r = _meter_to_pixel(_COURT_W, _COURT_L - _SERVICE_LINE, w, h)
    cv2.line(img, sl_bot_l, sl_bot_r, color, thickness, cv2.LINE_AA)

    # Center service line
    center_top = _meter_to_pixel(_COURT_W / 2.0, _SERVICE_LINE, w, h)
    center_bot = _meter_to_pixel(_COURT_W / 2.0, _COURT_L - _SERVICE_LINE, w, h)
    cv2.line(img, center_top, center_bot, color, thickness, cv2.LINE_AA)

    # Center marks on baselines
    cm1_a = _meter_to_pixel(_COURT_W / 2.0, 0, w, h)
    cm1_b = _meter_to_pixel(_COURT_W / 2.0, 0.3, w, h)
    cv2.line(img, cm1_a, cm1_b, color, thickness, cv2.LINE_AA)

    cm2_a = _meter_to_pixel(_COURT_W / 2.0, _COURT_L, w, h)
    cm2_b = _meter_to_pixel(_COURT_W / 2.0, _COURT_L - 0.3, w, h)
    cv2.line(img, cm2_a, cm2_b, color, thickness, cv2.LINE_AA)


def _render_heatmap_overlay(
    data: np.ndarray, w: int, h: int, player_id: int
) -> np.ndarray:
    """Render the heatmap data as a colored overlay image.

    Uses OpenCV's COLORMAP_JET and applies Gaussian blur for smooth
    density visualization.
    """
    # Transpose so x→columns, y→rows, then resize to image dimensions
    hmap = data.T.astype(np.float32)

    # Apply Gaussian blur for smooth density visualization
    # (kernel must be odd, scale to image size)
    kernel_size = max(int(min(w, h) * 0.08) | 1, 5)
    hmap = cv2.GaussianBlur(hmap, (kernel_size, kernel_size), 0)

    # Normalize to 0-255
    max_val = hmap.max()
    if max_val > 0:
        hmap = (hmap / max_val * 255).astype(np.uint8)
    else:
        hmap = np.zeros_like(hmap, dtype=np.uint8)

    # Resize to image dimensions (accounting for margins)
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.05)
    court_w_px = w - 2 * margin_x
    court_h_px = h - 2 * margin_y

    hmap_resized = cv2.resize(hmap, (court_w_px, court_h_px), interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    # Player 1: blue-hot (COLORMAP_JET), Player 2: magma-like (COLORMAP_INFERNO)
    colormap = cv2.COLORMAP_JET if player_id == 1 else cv2.COLORMAP_INFERNO
    hmap_colored = cv2.applyColorMap(hmap_resized, colormap)

    # Create full-size overlay with black background
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[margin_y:margin_y + court_h_px, margin_x:margin_x + court_w_px] = hmap_colored

    # Mask out areas with zero density (keep them black/transparent)
    mask = hmap_resized > 5  # Small threshold to avoid noise
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Create the base court background for non-heatmap areas
    bg = _draw_court_background(w, h)
    result = bg.copy()
    result[margin_y:margin_y + court_h_px, margin_x:margin_x + court_w_px][mask_3ch] = \
        hmap_colored[mask_3ch]

    return result


def _draw_title_and_legend(
    img: np.ndarray, player_id: int, img_w: int, max_density: float
) -> None:
    """Draw a title bar and color legend on the heatmap image."""
    color = config.P1_COLOR if player_id == 1 else config.P2_COLOR
    label = f"Player {player_id} — Court Heatmap"

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title background
    cv2.rectangle(img, (0, 0), (img_w, 35), (20, 20, 20), -1)
    cv2.putText(img, label, (10, 25), font, 0.7, color, 2, cv2.LINE_AA)

    # Legend bar at the bottom
    legend_h = 20
    legend_y = img.shape[0] - 40
    margin_x = int(img_w * 0.08)
    legend_w = img_w - 2 * margin_x

    # Draw gradient bar
    gradient = np.linspace(0, 255, legend_w).astype(np.uint8)
    gradient_bar = np.tile(gradient, (legend_h, 1))
    colormap = cv2.COLORMAP_JET if player_id == 1 else cv2.COLORMAP_INFERNO
    gradient_colored = cv2.applyColorMap(gradient_bar, colormap)
    img[legend_y:legend_y + legend_h, margin_x:margin_x + legend_w] = gradient_colored

    # Legend labels
    cv2.putText(img, "Low", (margin_x, legend_y + legend_h + 15),
                font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "High", (margin_x + legend_w - 30, legend_y + legend_h + 15),
                font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "Position Density", (img_w // 2 - 50, legend_y + legend_h + 15),
                font, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
