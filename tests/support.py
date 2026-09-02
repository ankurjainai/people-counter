"""Shared test helpers: tiny videos and a scripted detector."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from people_counter.detectors import Detection, Detector

WIDTH, HEIGHT, FPS, FRAMES = 320, 240, 25, 20


def write_blank_video(path: Path, frames: int = FRAMES, size=(WIDTH, HEIGHT), fps: int = FPS) -> Path:
    """A short, valid clip. Content is irrelevant when a stub detector is used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV has no writable codec in this environment")
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (10, 10 + i), (60, 90 + i), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


class ScriptedDetector(Detector):
    """Replays a fixed list of detections, one entry per processed frame."""

    name = "scripted"

    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self.script = list(script)
        self.calls = 0

    def detect(self, frame) -> List[Detection]:
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return list(self.script[index])


def straight_walk(x: float, y_from: float, y_to: float, steps: int, box=(40, 80)) -> List[List[Detection]]:
    """One person walking vertically: a per-frame detection script."""
    w, h = box
    script = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        y = y_from + (y_to - y_from) * t
        script.append([Detection(x - w / 2, y - h / 2, x + w / 2, y + h / 2, 0.9)])
    return script
