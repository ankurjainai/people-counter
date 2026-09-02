"""A small multi-object tracker: greedy association + short-term memory.

The pipeline needs identities, not just boxes: counting an entry means noticing
that *the same person* was on one side of the line a moment ago and is on the
other side now. This is a compact SORT-style tracker with linear motion
prediction in place of a Kalman filter — enough for shop-door footage, and with
no extra dependencies.

Association cost blends centroid distance (normalised by the frame diagonal) and
box IoU, so a track survives both a fast walker and a partially occluded one.
Re-attaching a track that missed a frame additionally demands that its
*predicted* box overlap the detection. Without that rule, a track left behind by
someone walking out of shot will happily capture the next person who appears
near the same spot — and since each track is only ever counted once, that person
would then cross the line uncounted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .detectors import Detection, iou


@dataclass
class Track:
    track_id: int
    box: Detection
    hits: int = 1
    age: int = 0                     # frames since the last matched detection
    total_frames: int = 1
    trail: List[Tuple[float, float]] = field(default_factory=list)
    counted: Optional[str] = None    # "in" | "out" | None
    last_side: Optional[float] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.box.centroid

    @property
    def confirmed(self) -> bool:
        return self.counted is not None

    def velocity(self, window: int = 5) -> Tuple[float, float]:
        """Average per-frame movement over the last few observations."""
        points = self.trail[-(window + 1):]
        if len(points) < 2:
            return (0.0, 0.0)
        (x0, y0), (x1, y1) = points[0], points[-1]
        steps = len(points) - 1
        return ((x1 - x0) / steps, (y1 - y0) / steps)

    def predicted_box(self, steps: int = 1) -> Detection:
        """Where this track should be ``steps`` frames after its last sighting."""
        if steps <= 0:
            return self.box
        vx, vy = self.velocity()
        dx, dy = vx * steps, vy * steps
        return Detection(
            self.box.x1 + dx, self.box.y1 + dy,
            self.box.x2 + dx, self.box.y2 + dy,
            self.box.score,
        )

    def update(self, box: Detection, max_trail: int = 48) -> None:
        self.box = box
        self.hits += 1
        self.age = 0
        self.total_frames += 1
        self.trail.append(box.centroid)
        if len(self.trail) > max_trail:
            del self.trail[: len(self.trail) - max_trail]

    def mark_missed(self) -> None:
        self.age += 1
        self.total_frames += 1


class CentroidTracker:
    def __init__(
        self,
        max_distance: float = 0.15,
        min_iou: float = 0.10,
        max_age: int = 30,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self.max_distance = max_distance
        self.min_iou = min_iou
        self.max_age = max_age
        self.set_frame_size(*frame_size)
        self._ids = itertools.count(1)
        self.tracks: Dict[int, Track] = {}

    def set_frame_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._diagonal = max(1.0, (width ** 2 + height ** 2) ** 0.5)

    # -- association -----------------------------------------------------
    def _cost(self, track: Track, det: Detection) -> Optional[float]:
        predicted = track.predicted_box(track.age)
        (tx, ty), (dx, dy) = predicted.centroid, det.centroid
        dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 / self._diagonal
        overlap = iou(predicted, det)

        if track.age > 0 and overlap < self.min_iou:
            # Re-attaching after a gap needs real evidence that this is the same
            # person, not merely someone standing where they used to be.
            return None

        # A generous overlap earns extra slack: people who jump between frames
        # (low fps, motion blur) still keep their identity.
        limit = self.max_distance * (2.0 if overlap >= self.min_iou else 1.0)
        if dist > limit:
            return None
        return dist - 0.5 * overlap

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Advance the tracker by one frame and return the live tracks."""
        pairs: List[Tuple[float, int, int]] = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                cost = self._cost(track, det)
                if cost is not None:
                    pairs.append((cost, tid, di))
        pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _cost, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            self.tracks[tid].update(detections[di])
            used_tracks.add(tid)
            used_dets.add(di)

        for tid, track in self.tracks.items():
            if tid not in used_tracks:
                track.mark_missed()

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            tid = next(self._ids)
            self.tracks[tid] = Track(track_id=tid, box=det, trail=[det.centroid])

        for tid in [t for t, tr in self.tracks.items() if tr.age > self.max_age]:
            del self.tracks[tid]

        return list(self.tracks.values())

    @property
    def active(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.age == 0]
