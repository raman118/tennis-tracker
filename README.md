# Tennis Player Tracking and Distance Measurement

A computer vision system that takes broadcast tennis footage and produces a fully annotated video with real-time distance tracking for both players. The pipeline detects players using YOLOv8x, establishes stable identities with ByteTrack, maps pixel positions to real-world meters through court homography, and accumulates distance traveled across the entire match.

Built with Python, OpenCV, and the Ultralytics YOLO framework.

---

## Features

- **Accurate player detection** using YOLOv8x with a multi-signal scoring system that distinguishes on-court players from ball boys, umpires, and spectators
- **Stable identity tracking** across thousands of frames with automatic re-identification recovery when tracking is temporarily lost
- **Real-world distance measurement** via four-point court homography, converting pixel displacement to meters using ITF standard court dimensions
- **Adaptive noise suppression** through EMA smoothing with variable alpha, minimum distance thresholds, and per-frame sanity caps to eliminate phantom movement from compression artifacts
- **Rich visual output** including bounding boxes, trajectory trails, a bird's-eye mini-court view, live speed indicators, detection quality bars, and a heads-up display panel
- **Post-match analytics engine** computing speed percentiles (avg, median, max, P95), movement phase classification (sprint / jog / walk / stationary), court zone occupancy, and court coverage percentage
- **Court heatmap generation** producing publication-quality PNG images showing position density for each player overlaid on an ITF-accurate court diagram
- **Three-stage court calibration** with automatic line detection, manual click fallback, and graceful pixel-only degradation
- **Performance profiling** with wall-clock time, average FPS throughput, and per-frame latency percentiles (P95, max) in every report
- **41 unit tests** covering core algorithms (EMA, distance, homography, scoring, analytics, config validation)
- **Batch processing** support for running the full pipeline across multiple videos sequentially
- **Structured JSON reports** with per-video metadata, full analytics breakdown, heatmap paths, and performance metrics

---

## Sample Output

The pipeline produces an annotated video with the following overlays on every frame:

```
+------------------------------------------+-----+
| [HUD Panel]                              |     |
|  Video: match_name                       |     |
|  Frame: 1200 / 4320                      |     |
|  Time:  0:40 / 2:24                      |     |
|  ----------------------                  |     |
|  P1 Distance: 47.23 m                    |     |
|  P2 Distance: 52.11 m                    |     |
|                                          |     |
|         +------+                         |     |
|         | P1   |  <-- blue bbox          |     |
|         | trail|     + label + trail     |     |
|         +--*---+     (* = foot point)    |     |
|                                          |     |
|                                          |     |
|              +------+                    |     |
|              | P2   |  <-- red bbox      |[   ]|
|              +--*---+                    |[   ]|
|                                          |[MC ]|
| [Stats Panel]                            |[   ]|
|  Speed   5.2 km/h    3.8 km/h           |[   ]|
|  Dist   47.2 m      52.1 m              +-----+
+------------------------------------------+
    MC = Mini-Court (bird's-eye view with player dots)
```

Each processed video also generates a JSON report:

```json
{
  "video_name": "tennis_video_assignment",
  "calibration_mode": "homography_auto",
  "total_frames": 27689,
  "fps": 119.88,
  "duration_seconds": 230.97,
  "player_1_distance_m": 721.73,
  "player_2_distance_m": 696.82,
  "analytics": {
    "player_1": {
      "speed": { "avg_kmh": 4.2, "median_kmh": 3.1, "max_kmh": 18.7, "p95_kmh": 11.3 },
      "movement_phases": {
        "stationary": { "pct": 42.1, "seconds": 97.2 },
        "walking":    { "pct": 28.5, "seconds": 65.8 },
        "jogging":    { "pct": 21.3, "seconds": 49.2 },
        "sprinting":  { "pct":  8.1, "seconds": 18.7 }
      },
      "court_zones": {
        "baseline_near_pct": 72.4,
        "net_zone_pct": 3.2
      },
      "court_coverage_pct": 34.7,
      "detection_quality": { "detection_rate_pct": 98.2 }
    },
    "player_2": { ... }
  },
  "heatmaps": {
    "player_1_heatmap": "outputs/tennis_video_assignment_heatmap_p1.png",
    "player_2_heatmap": "outputs/tennis_video_assignment_heatmap_p2.png"
  },
  "performance": {
    "wall_time_seconds": 245.3,
    "avg_fps": 112.8,
    "avg_frame_ms": 8.9,
    "p95_frame_ms": 14.2
  },
  "processed_at": "2026-02-28T13:57:13"
}
```

Court heatmaps are generated as standalone PNG images:

```
+----------------------------+   +----------------------------+
|  Player 1 — Court Heatmap  |   |  Player 2 — Court Heatmap  |
|  +-----------------------+ |   |  +-----------------------+ |
|  |       [baseline]      | |   |  |       [baseline]      | |
|  |   ████████            | |   |  |            ████████   | |
|  |   █HOT████            | |   |  |            ████HOT█   | |
|  |   ████████            | |   |  |            ████████   | |
|  |       --- net ---     | |   |  |       --- net ---     | |
|  |                       | |   |  |                       | |
|  |       [baseline]      | |   |  |       [baseline]      | |
|  +-----------------------+ |   |  +-----------------------+ |
|  [Low ■■■■■■■■■■■ High]   |   |  [Low ■■■■■■■■■■■ High]   |
+----------------------------+   +----------------------------+
```

---

## How It Works

### Pipeline Overview

```
Input Video
    |
    v
[1. Court Calibration]  -- Detect 4 court corners (auto/manual) --> Homography matrix
    |
    v
[2. Detection]          -- YOLOv8x + ByteTrack per frame --> Person detections with track IDs
    |
    v
[3. Filtering]          -- Multi-signal scoring (position, shape, size, confidence) --> Top 2 players
    |
    v
[4. Tracking]           -- EMA smoothing, ID locking, re-ID recovery --> Stable player states
    |
    v
[5. Distance]           -- Foot-point --> Homography --> Meters --> Threshold --> Accumulate
    |
    v
[6. Visualization]      -- Trails, boxes, HUD, mini-court, speed --> Annotated frame
    |
    v
[7. Analytics]          -- Speed stats, zones, coverage, heatmaps --> Rich JSON report
    |
    v
Output Video + JSON Report + Heatmap PNGs
```

### Stage Details

**Court Calibration.** On the first frame, the system runs Canny edge detection and Hough line transforms to find court lines, classifies them as horizontal (baselines) or vertical (sidelines), computes their intersections, and selects the four outermost corners. A homography matrix is then computed to map any pixel on the court plane to ITF-standard meter coordinates (8.23 m wide, 23.77 m long). If auto-detection fails, an interactive click window opens as fallback.

**Detection and Filtering.** Every frame is passed through YOLOv8x with ByteTrack for persistent tracking. Raw detections are scored by four weighted signals -- court zone position (40%), bounding box aspect ratio (25%), size relative to frame (20%), and YOLO confidence (15%) -- to separate real players from non-players. The top two scores are kept.

**Identity Locking.** During the first 30 frames (warmup), the system accumulates detection scores per track ID. The two IDs with the highest cumulative scores are locked as Player 1 and Player 2. Score-based locking ensures a consistently high-scoring player outranks a frequently-seen ball boy. After locking, a three-tier matching system handles each frame: direct ID match, velocity-gated nearest-centroid recovery for re-ID failures, and active re-locking when a player is lost for more than 5 consecutive frames.

**Distance Accumulation.** The foot-point (bottom-center of the bounding box) is smoothed via adaptive EMA, transformed to meters through the homography, and the Euclidean delta from the previous checkpoint is computed. Deltas below 0.05 m are suppressed as noise. Deltas above 1.5 m per frame are discarded as detection jumps. Valid deltas are added to the running total. The checkpoint only advances on accumulation, so slow movement below the per-frame threshold still accumulates once it crosses 0.05 m cumulatively.

**Post-Match Analytics.** After all frames are processed, the analytics engine computes comprehensive statistics from the full time-series of per-frame samples. Speed percentiles, movement phase classification (based on m/s thresholds), court zone occupancy, and coverage heatmaps are all derived in a single backward-looking pass. Court coverage is measured as the fraction of a 30×60 spatial bin grid that the player visited at least once.

---

## Project Structure

```
.
|-- main.py                     # Single-video CLI entry point
|-- process_all.py              # Batch processor for all videos in videos/
|-- requirements.txt            # Python dependencies
|-- LICENSE
|
|-- tennis_tracker/             # Core package
|   |-- __init__.py
|   |-- config.py               # All tunable constants (single source of truth)
|   |-- utils.py                # EMA, Euclidean distance, homography, file I/O
|   |-- calibration.py          # Court corner detection + homography computation
|   |-- detector.py             # YOLOv8x + ByteTrack + multi-signal scoring
|   |-- tracker.py              # ID locking, smoothing, distance accumulation
|   |-- visualizer.py           # All overlay rendering (trails, boxes, HUD)
|   |-- mini_court.py           # Bird's-eye court diagram overlay
|   |-- analytics.py            # Post-match analytics (speed, zones, coverage)
|   |-- heatmap.py              # Court heatmap image generation
|
|-- tests/                      # Unit test suite
|   |-- test_pipeline.py        # 41 tests covering core algorithms
|
|-- videos/                     # Input videos (not tracked by git)
|-- outputs/                    # Generated outputs (not tracked by git)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended for real-time throughput) or CPU

### Installation

```bash
git clone https://github.com/raman118/tennis-tracker.git
cd tennis-tracker

python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

YOLOv8x model weights are downloaded automatically on the first run.

### Quick Start

Place a video file in the `videos/` directory and run:

```bash
python main.py --input videos/your_match.mp4
```

The annotated video, JSON report, and heatmap images will appear in `outputs/`.

### Run Tests

```bash
python -m pytest tests/ -v
```

### All Commands

```bash
# Single video
python main.py --input videos/match.mp4

# Single video, headless (no GUI calibration fallback)
python main.py --input videos/match.mp4 --no-gui

# Single video, force CPU
python main.py --input videos/match.mp4 --device cpu

# Batch process every video in videos/
python process_all.py

# Batch, headless
python process_all.py --no-gui --device cuda
```

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to input video | Interactive selection prompt |
| `--output` | Path for annotated output video | Auto-generated in `outputs/` |
| `--no-gui` | Skip manual calibration window | GUI enabled |
| `--device` | Force `cuda` or `cpu` | Auto-detect |

---

## Technical Design

### Why YOLOv8x

The extra-large variant has the highest mAP on COCO among YOLOv8 models. Players at the far baseline can occupy fewer than 40 vertical pixels in broadcast footage, making detection accuracy critical. Since the pipeline processes video offline, the speed penalty relative to smaller variants is acceptable.

### Why ByteTrack

ByteTrack performs Kalman filter prediction with IoU-based association. It requires no separate re-identification model, adds no extra dependencies (built into ultralytics), and excels on fixed-camera footage where inter-frame motion is smooth and predictable.

### Why Foot-Point Instead of BBox Center

The bounding box center sits at the player's torso, roughly 0.9 m above the court. The homography maps the court *plane*, so projecting an elevated point introduces parallax error proportional to camera distance. The foot-point (bottom-center of the bbox) lies on the court surface and is geometrically correct for the transform.

### Why Adaptive EMA

Compression artifacts cause 2-4 pixel jitter per frame. Over thousands of frames, this accumulates to 70-140 m of phantom distance. A fixed EMA alpha is either too sluggish for sprints or too noisy when stationary. The adaptive approach uses alpha = 0.6 during fast movement, 0.4 for medium, and 0.15 when nearly stationary, paired with a 0.05 m noise floor.

### Why Homography Instead of Pixel Scaling

Broadcast cameras are angled at 30-45 degrees, creating perspective distortion where near-camera objects appear larger. A linear pixel-to-meter constant cannot correct for this. The four-point homography provides a full projective transformation that handles arbitrary camera angles as long as the four court corners are known.

### Why Score-Based ID Locking

Frequency-based locking (most-seen track ID = player) fails when a ball boy near the sideline is detected in more frames than a far-baseline player. Cumulative score-based locking weights both frequency and detection quality, so a player scoring 0.9 per frame always outranks a ball boy scoring 0.2.

### Why Post-Processing Analytics Instead of Per-Frame

Computing statistics like speed percentiles, movement phase breakdowns, and coverage heatmaps requires the full time-series of samples. Attempting these in a streaming per-frame model would either require awkward windowing or miss backward-looking computations entirely. The analytics engine collects lightweight per-frame samples (tuple of position + speed + detected flag) during the main loop, then computes everything in a single pass afterward. The memory overhead is negligible (~50 bytes per frame × 30,000 frames ≈ 1.5 MB).

---

## Configuration

All parameters live in `tennis_tracker/config.py` with inline documentation. Key values:

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8x.pt` | Model weights file |
| `CONF_THRESHOLD` | `0.5` | Minimum YOLO confidence to keep a detection |
| `WARMUP_FRAMES` | `30` | Observation window before locking player IDs |
| `EMA_ALPHA` | `0.3` | Base smoothing factor (overridden adaptively) |
| `MIN_DIST_THRESHOLD_M` | `0.05` | Noise floor -- ignore movement below this |
| `MAX_DELTA_PER_FRAME_M` | `1.5` | Sanity cap -- discard impossible single-frame jumps |
| `COURT_WIDTH_M` | `8.23` | ITF singles court width in meters |
| `COURT_LENGTH_M` | `23.77` | ITF singles court length in meters |
| `TRAIL_LENGTH` | `60` | Trajectory trail length in frames (~2 sec at 30 fps) |
| `COURT_MARGIN_PERCENT` | `0.15` | How much to expand court polygon for boundary checks |
| `MAX_RECOVERY_DIST_PX` | `200` | Max pixel distance for nearest-centroid re-ID |
| `MAX_PIXELS_PER_FRAME` | `80` | Velocity gate -- reject re-ID beyond this displacement |

---

## Court Calibration

The calibration follows a three-stage cascade so the pipeline always produces output:

**Stage 1 -- Automatic.** Canny edges, Hough lines, angle classification, intersection computation, quadrant selection, and aspect ratio validation against known court proportions. Works on clean broadcast footage with visible white court lines.

**Stage 2 -- Manual.** If auto-detection fails and `--no-gui` is not set, a window opens showing frame 0. Click the four singles court corners in order: top-left, top-right, bottom-right, bottom-left. Press `r` to reset. The window closes after the fourth click.

**Stage 3 -- Pixel fallback.** If both fail, distances are reported in pixels instead of meters. The JSON report indicates `"calibration_mode": "pixel_fallback"`.

---

## Limitations

- **Static camera only.** Homography is computed once from frame 0. Camera movement (pan, zoom, cuts) will invalidate the calibration.
- **Broadcast overlays persist.** Score graphics and network logos burned into the source frames are not removed.
- **Edge-of-frame degradation.** Players near the frame boundary have clipped bounding boxes, reducing foot-point accuracy.
- **Close-proximity ID swaps.** When both players are within ~1 m (e.g., net exchanges), ByteTrack may briefly swap identities.
- **First-frame dependency.** Auto-calibration requires court lines to be clearly visible in frame 0. Overlay graphics or exposure changes in the opening frame will trigger manual fallback.

---

## Dependencies

- [ultralytics](https://github.com/ultralytics/ultralytics) -- YOLOv8 + ByteTrack
- [OpenCV](https://opencv.org/) -- Video I/O, image processing, homography
- [NumPy](https://numpy.org/) -- Array operations
- [PyTorch](https://pytorch.org/) -- Backend for YOLO inference
- [tqdm](https://github.com/tqdm/tqdm) -- Progress bars

---

## License

Released under the [MIT License](LICENSE).
