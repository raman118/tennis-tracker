"""
Main entry point for the Tennis Player Tracking pipeline.

Processes a single tennis match video through the full pipeline:
  Detection → Tracking → Calibration → Distance accumulation →
  Annotated video output + JSON report + console summary.

Usage:
  python main.py --input videos/tennis_video_assignment.mp4
  python main.py --input videos/tennis_video_assignment.mp4 --no-gui
  python main.py --input videos/tennis_video_assignment.mp4 --device cpu

Architecture decision: the pipeline is exposed as both a CLI application
(via argparse) and a callable function (run_pipeline) so that process_all.py
can invoke it programmatically without subprocess overhead.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
from tqdm import tqdm

from tennis_tracker import config
from tennis_tracker.analytics import AnalyticsEngine
from tennis_tracker.calibration import CourtCalibrator
from tennis_tracker.detector import PlayerDetector
from tennis_tracker.heatmap import generate_heatmap_image
from tennis_tracker.tracker import PlayerTracker
from tennis_tracker.utils import euclidean_dist
from tennis_tracker.visualizer import Visualizer

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    interactive: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """Execute the full tennis tracking pipeline on a single video.

    This is the core function that orchestrates all pipeline stages. It is
    called directly by main() for single-video CLI mode and by process_all.py
    for batch processing.

    Args:
        input_path: Path to the input video file.
        output_path: Path for the output annotated video. If None, auto-generated
                     from input stem following the naming convention.
        interactive: If True, allow GUI-based manual court calibration fallback.
        device: Override device ('cuda' or 'cpu'). None = use config default.

    Returns:
        Dictionary containing the JSON report data.

    Raises:
        FileNotFoundError: If the input video cannot be opened.
    """
    # --- Resolve paths ---
    input_file = Path(input_path).resolve()
    video_stem = input_file.stem

    if not input_file.is_file():
        raise FileNotFoundError(f"Input video not found: {input_file}")

    # Auto-generate output paths if not explicitly provided
    # Resolve relative to the project root (parent of 'videos/' directory)
    project_root = input_file.parent.parent if input_file.parent.name == config.VIDEOS_DIR else input_file.parent
    outputs_dir = project_root / config.OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_video = outputs_dir / f"{video_stem}_output.mp4"
    else:
        output_video = Path(output_path).resolve()

    report_path = outputs_dir / f"{video_stem}_report.json"

    logger.info("Input  : %s", input_file)
    logger.info("Output : %s", output_video)
    logger.info("Report : %s", report_path)

    # --- Override device if requested ---
    if device is not None:
        config.DEVICE = device
        logger.info("Device overridden to: %s", device)

    # --- Open video ---
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cv2.VideoCapture failed to open: {input_file}")

    # Extract video properties for output writer and HUD display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(fps, 1.0)

    logger.info(
        "Video opened: %dx%d, %.1f fps, %d frames, duration %.1f sec.",
        width, height, fps, total_frames, duration_sec,
    )

    # --- Read first frame for court calibration ---
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        raise FileNotFoundError(f"Could not read first frame from: {input_file}")

    # Reset video to beginning so the main loop starts from frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Initialize pipeline components ---
    calibrator = CourtCalibrator(first_frame, interactive=interactive)

    detector = PlayerDetector()
    tracker = PlayerTracker(calibrator)
    visualizer = Visualizer()
    analytics = AnalyticsEngine(fps=fps, total_frames=total_frames)

    # --- Initialize video writer ---
    # mp4v (MPEG-4 Part 2) codec is universally supported and produces
    # reasonable quality at ~2-3 MB/min for 720p broadcast content.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for: %s", output_video)
        cap.release()
        raise IOError(f"VideoWriter failed to open: {output_video}")

    logger.info("Processing %d frames...", total_frames)

    # --- Performance tracking ---
    wall_start = time.perf_counter()
    frame_times = []

    # --- Main processing loop ---
    # tqdm provides a progress bar with ETA and throughput (it/s)
    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_stem}", unit="frame"):
        frame_t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            # Premature end of video — can happen with corrupted files
            logger.warning("Frame read failed at index %d — stopping early.", frame_idx)
            break

        # Step 1: Detect all persons in this frame (with ByteTrack tracking)
        detections = detector.detect(frame)

        # Step 2: Filter to the two on-court players (exclude ball boys etc.)
        # After warmup, pass locked track IDs so filter prioritizes real players
        # over higher-confidence non-players (ball boys near the court edge).
        locked_ids = None
        if tracker.ids_locked:
            p1_state, p2_state = tracker.get_states()
            locked_ids = (p1_state.locked_track_id, p2_state.locked_track_id)
        filtered = detector.filter_to_players(detections, calibrator, locked_ids, frame.shape)

        # Step 3: Update tracker state (smoothing, distance accumulation)
        tracker.update(filtered, frame_idx, frame.shape)

        # Step 4: Get current player states for visualization
        p1, p2 = tracker.get_states()

        # Step 4b: Record analytics sample for each player
        for pid, ps in [(1, p1), (2, p2)]:
            speed_mps = 0.0
            if ps.meter_position and ps.prev_meter_pos and fps > 0:
                speed_mps = euclidean_dist(ps.meter_position, ps.prev_meter_pos) * fps
            analytics.record(
                player_id=pid,
                frame_idx=frame_idx,
                meter_pos=ps.meter_position,
                speed_mps=speed_mps,
                detected=ps.detected_this_frame,
            )

        # Step 5: Render all overlays onto the frame
        annotated = visualizer.render(
            frame, p1, p2, frame_idx, total_frames, fps, video_stem,
        )

        # Step 6: Write annotated frame to output video
        writer.write(annotated)

        # Step 7: Record frame processing time
        frame_times.append(time.perf_counter() - frame_t0)

    # --- Release resources ---
    cap.release()
    writer.release()
    wall_elapsed = time.perf_counter() - wall_start
    logger.info("Video processing complete. Output saved to: %s", output_video)

    # --- Final player states ---
    p1, p2 = tracker.get_states()

    # --- Compute analytics ---
    match_analytics = analytics.compute(
        p1_distance=p1.total_distance_m,
        p2_distance=p2.total_distance_m,
    )
    analytics_dict = AnalyticsEngine.analytics_to_dict(match_analytics)

    # --- Generate heatmap images ---
    heatmap_paths = {}
    for pid, pa in [(1, match_analytics.player_1), (2, match_analytics.player_2)]:
        hmap_path = outputs_dir / f"{video_stem}_heatmap_p{pid}.png"
        saved = generate_heatmap_image(pa.heatmap, pid, str(hmap_path))
        if saved:
            heatmap_paths[f"player_{pid}_heatmap"] = str(hmap_path)

    # --- Performance metrics ---
    import numpy as _np
    ft = _np.array(frame_times) if frame_times else _np.array([0.0])
    perf_metrics = {
        "wall_time_seconds": round(wall_elapsed, 2),
        "avg_fps": round(len(frame_times) / max(wall_elapsed, 0.001), 1),
        "avg_frame_ms": round(float(_np.mean(ft)) * 1000, 1),
        "p95_frame_ms": round(float(_np.percentile(ft, 95)) * 1000, 1),
        "max_frame_ms": round(float(_np.max(ft)) * 1000, 1),
    }

    # --- Build JSON report ---
    report = {
        "video_name": video_stem,
        "input_path": str(input_file),
        "output_path": str(output_video),
        "calibration_mode": calibrator.calibration_mode,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": round(duration_sec, 2),
        "player_1_distance_m": round(p1.total_distance_m, 2),
        "player_2_distance_m": round(p2.total_distance_m, 2),
        **analytics_dict,
        "heatmaps": heatmap_paths,
        "performance": perf_metrics,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Save JSON report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("JSON report saved to: %s", report_path)

    # --- Print formatted console summary ---
    _print_summary(report, str(output_video), str(report_path))

    return report


def _print_summary(report: Dict, output_path: str, report_path: str) -> None:
    """Print a formatted final summary table to the console.

    This is the one place where bare print() is used intentionally
    (not logging) because the summary is user-facing output meant to
    be visually distinct from log messages.

    Args:
        report: The JSON report dictionary.
        output_path: Path to the output video.
        report_path: Path to the JSON report file.
    """
    from tennis_tracker.utils import format_time

    duration_str = format_time(report["duration_seconds"])
    video_name = report["video_name"]
    frames = report["total_frames"]
    cal_mode = report["calibration_mode"]
    p1_dist = report["player_1_distance_m"]
    p2_dist = report["player_2_distance_m"]

    # Extract analytics sub-dict safely
    a = report.get("analytics", {})
    p1a = a.get("player_1", {})
    p2a = a.get("player_2", {})
    p1_spd = p1a.get("speed", {})
    p2_spd = p2a.get("speed", {})
    perf = report.get("performance", {})

    # Box-drawing characters for a professional-looking console table
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║          TENNIS TRACKER — COMPLETE                 ║")
    print("╠════════════════════════════════════════════════════╣")
    print(f"║  Video      : {video_name:<36s}║")
    print(f"║  Frames     : {frames:<6d} |  Duration : {duration_str:<15s}║")
    print(f"║  Calibration: {cal_mode:<36s}║")
    print("║  ────────────────────────────────────────────────  ║")
    print(f"║  Player 1   : {p1_dist:<8.2f} m   avg {p1_spd.get('avg_kmh', 0):>5.1f} km/h     ║")
    print(f"║  Player 2   : {p2_dist:<8.2f} m   avg {p2_spd.get('avg_kmh', 0):>5.1f} km/h     ║")
    print(f"║  P1 Coverage: {p1a.get('court_coverage_pct', 0):>5.1f}%   P2 Coverage: {p2a.get('court_coverage_pct', 0):>5.1f}%   ║")
    print("║  ────────────────────────────────────────────────  ║")
    print(f"║  Throughput : {perf.get('avg_fps', 0):>6.1f} fps  ({perf.get('avg_frame_ms', 0):.1f} ms/frame)      ║")
    print(f"║  Wall time  : {perf.get('wall_time_seconds', 0):>6.1f} sec                         ║")
    print("║  ────────────────────────────────────────────────  ║")
    print(f"║  Output     : {Path(output_path).name:<36s}║")
    print(f"║  Report     : {Path(report_path).name:<36s}║")
    print("╚════════════════════════════════════════════════════╝")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Tennis Player Tracking & Distance Measurement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input video file. If omitted, prompts for selection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output annotated video (default: auto-generated).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI-based manual calibration fallback (auto-detect only).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Force inference device. Default: auto-detect (CUDA if available).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for single-video processing."""
    args = parse_args()

    # If no input specified, list available videos and prompt the user
    if args.input is None:
        from tennis_tracker.utils import get_video_files

        videos = get_video_files(config.VIDEOS_DIR)
        if not videos:
            logger.error("No video files found in '%s/'.", config.VIDEOS_DIR)
            sys.exit(1)

        print("\nAvailable videos:")
        for i, v in enumerate(videos, 1):
            print(f"  {i}. {v.name}")

        try:
            choice = int(input("\nSelect video number: ")) - 1
            if 0 <= choice < len(videos):
                input_path = str(videos[choice])
            else:
                logger.error("Invalid selection: %d", choice + 1)
                sys.exit(1)
        except (ValueError, EOFError):
            logger.error("Invalid input. Please enter a number.")
            sys.exit(1)
    else:
        input_path = args.input

    run_pipeline(
        input_path=input_path,
        output_path=args.output,
        interactive=not args.no_gui,
        device=args.device,
    )


if __name__ == "__main__":
    main()
