"""End-to-end video processing: detect -> track -> count -> annotate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from . import annotate
from .config import CountingConfig
from .counter import Counts, LineCounter
from .detectors import Detector, build_detector
from .tracker import CentroidTracker

log = logging.getLogger(__name__)

ProgressFn = Callable[["Progress"], None]

# OpenCV's FFMPEG backend writes avc1 (H.264) on most builds, which every
# browser can play inline. mp4v is the last-resort fallback.
_CODECS = ("avc1", "mp4v")


@dataclass
class Progress:
    frames_done: int
    frames_total: Optional[int]
    entered: int
    exited: int
    elapsed: float

    @property
    def fraction(self) -> Optional[float]:
        if not self.frames_total:
            return None
        return min(1.0, self.frames_done / self.frames_total)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: Optional[int]

    @property
    def duration(self) -> Optional[float]:
        if not self.frame_count or self.fps <= 0:
            return None
        return self.frame_count / self.fps

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration": round(self.duration, 2) if self.duration else None,
        }


@dataclass
class CountResult:
    counts: Counts
    video: VideoInfo
    detector: str
    frames_processed: int
    elapsed: float
    output_path: Optional[str] = None
    config: Dict = field(default_factory=dict)

    @property
    def fps_processed(self) -> float:
        return self.frames_processed / self.elapsed if self.elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            **self.counts.to_dict(),
            "video": self.video.to_dict(),
            "detector": self.detector,
            "frames_processed": self.frames_processed,
            "elapsed": round(self.elapsed, 2),
            "processing_fps": round(self.fps_processed, 2),
            "output_path": self.output_path,
            "config": self.config,
        }

    def summary(self) -> str:
        v = self.video
        return (
            f"Entered: {self.counts.entered}\n"
            f"Exited:  {self.counts.exited}\n"
            f"Net occupancy change: {self.counts.occupancy:+d}\n"
            f"Detector: {self.detector} | {v.width}x{v.height} @ {v.fps:.1f} fps\n"
            f"Processed {self.frames_processed} frames in {self.elapsed:.1f}s "
            f"({self.fps_processed:.1f} fps)"
        )


def probe(path: str | Path) -> VideoInfo:
    """Read a video's dimensions without decoding it fully."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=count if count > 0 else None,
        )
    finally:
        cap.release()


def grab_frame(path: str | Path, index: int = 0):
    """Return a single BGR frame — used for the front-end's line-drawing preview."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        if index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {index} from {path}")
        return frame
    finally:
        cap.release()


def _open_writer(path: Path, fps: float, size) -> cv2.VideoWriter:
    last_error = None
    for codec in _CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            log.debug("Writing %s with codec %s", path, codec)
            return writer
        writer.release()
        last_error = codec
    raise RuntimeError(f"No usable video codec (last tried {last_error}) for {path}")


def count_people(
    video_path: str | Path,
    config: Optional[CountingConfig] = None,
    output_path: Optional[str | Path] = None,
    progress: Optional[ProgressFn] = None,
    progress_every: int = 10,
    detector: Optional[Detector] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> CountResult:
    """Count people entering and exiting across ``config.line``.

    ``detector`` may be supplied to reuse a loaded model across runs; otherwise
    one is built from ``config``. ``cancelled`` is polled once per processed
    frame so a web request can abort a long job.
    """
    config = config or CountingConfig()
    info = probe(video_path)
    detector = detector or build_detector(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    stride = max(1, int(config.frame_stride))
    tracker = CentroidTracker(
        max_distance=config.max_distance,
        min_iou=config.min_iou,
        max_age=config.max_age,
        frame_size=(info.width, info.height),
    )
    counter = LineCounter(config, info.width, info.height)

    writer = None
    out_path: Optional[Path] = None
    if config.write_video and output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The output keeps real time: with a stride, fewer frames at lower fps.
        writer = _open_writer(out_path, max(1.0, info.fps / stride), (info.width, info.height))

    total = info.frame_count
    if total and config.max_frames:
        total = min(total, config.max_frames * stride)
    frames_total = (total // stride) if total else None

    started = time.time()
    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            if config.max_frames and processed >= config.max_frames:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            timestamp = frame_index / info.fps if info.fps > 0 else 0.0
            counter.update(tracks, frame_index, timestamp)
            processed += 1

            if writer is not None:
                annotate.draw_line(frame, config.line)
                annotate.draw_tracks(frame, tracks, config.draw_trails)
                annotate.draw_hud(
                    frame,
                    counter.counts,
                    extra=[f"t={timestamp:6.2f}s  frame {frame_index}"],
                )
                writer.write(frame)

            if progress and processed % progress_every == 0:
                progress(
                    Progress(
                        processed,
                        frames_total,
                        counter.counts.entered,
                        counter.counts.exited,
                        time.time() - started,
                    )
                )
            if cancelled is not None and cancelled():
                log.info("Run cancelled after %d frames", processed)
                break
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - started
    if progress:
        progress(
            Progress(
                processed, frames_total, counter.counts.entered, counter.counts.exited, elapsed
            )
        )

    return CountResult(
        counts=counter.counts,
        video=info,
        detector=detector.name,
        frames_processed=processed,
        elapsed=elapsed,
        output_path=str(out_path) if out_path else None,
        config={
            "line": config.line.to_dict(),
            "confidence": config.confidence,
            "frame_stride": stride,
            "min_hits": config.min_hits,
            "max_age": config.max_age,
        },
    )
