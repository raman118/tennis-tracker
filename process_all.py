"""
Batch processor for the Tennis Player Tracking pipeline.

Scans the videos/ directory for all valid video files and runs the full
tracking pipeline on each. After all videos are processed, prints a
combined summary table.

Usage:
  python process_all.py
  python process_all.py --no-gui
  python process_all.py --device cpu

Architecture decision: this module reuses main.run_pipeline() directly
in-process rather than launching subprocesses. This avoids model reload
overhead (YOLOv8x takes ~5 seconds to load) and shares GPU memory efficiently.
However, ByteTrack's internal state is tied to the YOLO model instance, so
we re-initialize the pipeline per video to avoid cross-video tracker contamination.
"""

import argparse
import logging
import sys
from typing import Dict, List

from tennis_tracker import config
from tennis_tracker.utils import get_video_files
from main import run_pipeline

logger = logging.getLogger(__name__)


def process_all(interactive: bool = True, device: str = None) -> List[Dict]:
    """Process all video files in the videos/ directory.

    Args:
        interactive: If True, allow GUI calibration fallback per video.
        device: Override device ('cuda' or 'cpu'). None = use config default.

    Returns:
        List of JSON report dictionaries, one per video.
    """
    videos = get_video_files(config.VIDEOS_DIR)

    if not videos:
        logger.error("No video files found in '%s/'.", config.VIDEOS_DIR)
        sys.exit(1)

    logger.info("Found %d video(s) to process.", len(videos))

    reports: List[Dict] = []

    for i, video_path in enumerate(videos, 1):
        logger.info("━━━ Processing video %d/%d: %s ━━━", i, len(videos), video_path.name)

        try:
            report = run_pipeline(
                input_path=str(video_path),
                output_path=None,         # Auto-generate output path
                interactive=interactive,
                device=device,
            )
            reports.append(report)
        except Exception as e:
            logger.error("Failed to process '%s': %s", video_path.name, e)
            # Continue to next video rather than aborting the entire batch
            continue

    # Print combined summary table
    if reports:
        _print_combined_summary(reports)

    return reports


def _print_combined_summary(reports: List[Dict]) -> None:
    """Print a formatted summary table showing all processed videos.

    Uses box-drawing characters for a clean, professional console output
    that is easy to scan at a glance.

    Args:
        reports: List of per-video JSON report dictionaries.
    """
    # Column widths — accommodating long video names
    col_video = 33
    col_p1 = 10
    col_p2 = 10

    # Table header
    print()
    print("┌" + "─" * col_video + "┬" + "─" * col_p1 + "┬" + "─" * col_p2 + "┐")
    print(
        "│"
        + " Video".ljust(col_video)
        + "│"
        + " P1 (m)".ljust(col_p1)
        + "│"
        + " P2 (m)".ljust(col_p2)
        + "│"
    )
    print("├" + "─" * col_video + "┼" + "─" * col_p1 + "┼" + "─" * col_p2 + "┤")

    # Table rows
    for r in reports:
        name = r["video_name"]
        p1 = f'{r["player_1_distance_m"]:.2f}'
        p2 = f'{r["player_2_distance_m"]:.2f}'
        print(
            "│"
            + f" {name}".ljust(col_video)
            + "│"
            + f" {p1}".ljust(col_p1)
            + "│"
            + f" {p2}".ljust(col_p2)
            + "│"
        )

    # Table footer
    print("└" + "─" * col_video + "┴" + "─" * col_p1 + "┴" + "─" * col_p2 + "┘")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch processing.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Batch process all tennis videos in the videos/ directory.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI-based manual calibration fallback.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Force inference device. Default: auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for batch processing."""
    args = parse_args()
    process_all(interactive=not args.no_gui, device=args.device)


if __name__ == "__main__":
    main()
