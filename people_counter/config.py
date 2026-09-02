"""Configuration objects for the people-counting pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass(frozen=True)
class CountingLine:
    """A virtual tripwire, expressed in *normalised* frame coordinates (0..1).

    Normalised coordinates let the front-end draw the line on a preview image of
    any size and have it mean the same thing at full video resolution.

    ``inside_is_left`` decides which side of the directed segment (x1,y1)->(x2,y2)
    counts as "inside the shop". With the default, a person crossing from the
    right-hand side to the left-hand side of the arrow is counted as an ENTRY.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    inside_is_left: bool = True

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"CountingLine.{name} must be in [0, 1], got {v!r}")
        if math.isclose(self.x1, self.x2) and math.isclose(self.y1, self.y2):
            raise ValueError("CountingLine endpoints must differ")

    def pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )

    def side(self, px: float, py: float, width: int, height: int) -> float:
        """Signed area of the triangle (p1, p2, point).

        Positive => the point lies to the left of the directed segment.
        """
        x1, y1, x2, y2 = self.pixels(width, height)
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CountingLine":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            inside_is_left=bool(data.get("inside_is_left", True)),
        )

    @classmethod
    def horizontal(cls, y: float = 0.5, inside_is_left: bool = True) -> "CountingLine":
        """A horizontal line across the frame at height ``y``.

        Image coordinates put y=0 at the top, so the left half-plane of a
        segment drawn right-to-left is the region *above* it. Building the line
        that way makes the intuitive default true for doorway footage: with
        ``inside_is_left``, walking up the frame and away from the camera is an
        entry, and walking down towards the camera is an exit.
        """
        return cls(1.0, y, 0.0, y, inside_is_left)


@dataclass
class CountingConfig:
    """Everything that steers a single counting run."""

    line: CountingLine = field(default_factory=CountingLine.horizontal)

    # --- detector -------------------------------------------------------
    detector: str = "auto"          # auto | yolo | motion | hog
    model: str = "yolov8n.pt"       # ultralytics weights (yolo detector only)
    confidence: float = 0.35
    imgsz: int = 640
    device: Optional[str] = None    # None -> ultralytics picks (cpu/mps/cuda)

    # --- tracker --------------------------------------------------------
    max_distance: float = 0.15      # max centroid jump per frame, as a fraction
                                    # of the frame diagonal
    min_iou: float = 0.10           # box overlap that can rescue a big jump
    max_age: int = 30               # frames a track survives without a detection
    min_hits: int = 3               # detections before a track may count

    # --- sampling / output ---------------------------------------------
    frame_stride: int = 1           # process every Nth frame
    max_frames: Optional[int] = None
    write_video: bool = True
    draw_trails: bool = True

    def resolved_detector(self) -> str:
        if self.detector != "auto":
            return self.detector
        try:
            import ultralytics  # noqa: F401
        except Exception:
            return "motion"
        return "yolo"
