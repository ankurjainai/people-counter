"""Overlay drawing for the annotated output video."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from .config import CountingLine
from .counter import Counts
from .tracker import Track

GREEN = (80, 220, 100)
RED = (70, 80, 240)
BLUE = (240, 180, 70)
WHITE = (255, 255, 255)
DARK = (28, 28, 32)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color_for(track: Track) -> Tuple[int, int, int]:
    if track.counted == "in":
        return GREEN
    if track.counted == "out":
        return RED
    return BLUE


def draw_line(frame: np.ndarray, line: CountingLine) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = line.pixels(w, h)
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y2), (60, 200, 255), 2, cv2.LINE_AA)

    # An arrow on the "inside" side of the midpoint shows which way is an entry.
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    # Unit normal pointing into the left half-plane, i.e. the side where
    # CountingLine.side() is positive.
    nx, ny = -dy / length, dx / length
    if not line.inside_is_left:
        nx, ny = -nx, -ny
    tip = (int(mx + nx * 46), int(my + ny * 46))
    cv2.arrowedLine(frame, (int(mx), int(my)), tip, GREEN, 3, cv2.LINE_AA, tipLength=0.35)
    # Sit the label clear of the arrow head, along the arrow and off to one side.
    _label(frame, "IN", (int(mx + nx * 60 - ny * 12), int(my + ny * 60 + nx * 12) + 5), GREEN)


def draw_tracks(frame: np.ndarray, tracks: Iterable[Track], draw_trails: bool = True) -> None:
    for track in tracks:
        if track.age != 0:
            continue
        color = _color_for(track)
        x1, y1, x2, y2 = track.box.as_ints()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _label(frame, f"#{track.track_id}", (x1, max(14, y1 - 6)), color)

        cx, cy = track.centroid
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1, cv2.LINE_AA)

        if draw_trails and len(track.trail) > 1:
            pts = np.array([[int(x), int(y)] for x, y in track.trail], dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, counts: Counts, extra: Sequence[str] = ()) -> None:
    lines = [
        (f"IN  {counts.entered}", GREEN),
        (f"OUT {counts.exited}", RED),
        (f"NET {counts.occupancy}", WHITE),
    ]
    pad, line_h = 12, 30
    box_h = pad * 2 + line_h * (len(lines) + len(extra))
    box_w = 190

    panel = frame[pad : pad + box_h, pad : pad + box_w]
    if panel.size:
        frame[pad : pad + box_h, pad : pad + box_w] = cv2.addWeighted(
            panel, 0.35, np.full_like(panel, DARK, dtype=np.uint8), 0.65, 0
        )

    y = pad + 24
    for text, color in lines:
        cv2.putText(frame, text, (pad + 12, y), _FONT, 0.72, color, 2, cv2.LINE_AA)
        y += line_h
    for text in extra:
        cv2.putText(frame, text, (pad + 12, y), _FONT, 0.5, (200, 200, 205), 1, cv2.LINE_AA)
        y += line_h


def _label(frame: np.ndarray, text: str, origin: Tuple[int, int], color) -> None:
    cv2.putText(frame, text, origin, _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _FONT, 0.55, color, 1, cv2.LINE_AA)
