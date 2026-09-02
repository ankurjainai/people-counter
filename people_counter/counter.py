"""Line-crossing logic: turn tracks into entry / exit counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .config import CountingConfig, CountingLine
from .tracker import Track


@dataclass
class CrossingEvent:
    track_id: int
    direction: str          # "in" | "out"
    frame_index: int
    timestamp: float        # seconds into the clip
    x: float
    y: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "direction": self.direction,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
        }


@dataclass
class Counts:
    entered: int = 0
    exited: int = 0
    events: List[CrossingEvent] = field(default_factory=list)

    @property
    def occupancy(self) -> int:
        """Net people inside, assuming the room started empty."""
        return self.entered - self.exited

    def to_dict(self) -> dict:
        return {
            "entered": self.entered,
            "exited": self.exited,
            "occupancy": self.occupancy,
            "events": [e.to_dict() for e in self.events],
        }


class LineCounter:
    """Counts a track the first time its centroid changes side of the line.

    A track must have been seen ``min_hits`` times before it may count, which
    suppresses one-frame detector flickers near the doorway. Each track is
    counted at most once, so a person loitering on the threshold cannot inflate
    the totals.
    """

    def __init__(self, config: CountingConfig, width: int, height: int) -> None:
        self.line: CountingLine = config.line
        self.min_hits = config.min_hits
        self.width = width
        self.height = height
        self.counts = Counts()

    def _direction(self, previous: float, current: float) -> Optional[str]:
        if (previous > 0) == (current > 0):
            return None
        # Moving into the positive half-plane == moving to the *left* of the
        # directed segment.
        moved_left = current > 0
        inside = self.line.inside_is_left
        return "in" if moved_left == inside else "out"

    def update(self, tracks: Iterable[Track], frame_index: int, timestamp: float) -> List[CrossingEvent]:
        fired: List[CrossingEvent] = []
        for track in tracks:
            if track.age != 0:
                continue  # only judge tracks backed by a detection this frame
            cx, cy = track.centroid
            side = self.line.side(cx, cy, self.width, self.height)
            if side == 0.0:
                # Dead on the line: neither side yet. Keep the last decided
                # side so the crossing is still seen on the following frame.
                continue
            previous, track.last_side = track.last_side, side
            if previous is None or track.counted is not None:
                continue
            if track.hits < self.min_hits:
                continue
            direction = self._direction(previous, side)
            if direction is None:
                continue
            track.counted = direction
            event = CrossingEvent(track.track_id, direction, frame_index, timestamp, cx, cy)
            if direction == "in":
                self.counts.entered += 1
            else:
                self.counts.exited += 1
            self.counts.events.append(event)
            fired.append(event)
        return fired
