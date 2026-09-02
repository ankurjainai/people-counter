#!/usr/bin/env python3
"""Render a synthetic doorway clip with a known ground truth.

Useful for smoke-testing the whole pipeline without hunting for real footage:
each "person" is a walking figure with a head and body, large enough for the
HOG/YOLO detectors to see, moving on a straight path across the frame.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 960, 540, 25


def _draw_person(frame: np.ndarray, cx: float, cy: float, height: int, phase: float, shade) -> None:
    """Draw a walking figure with roughly human proportions.

    Detectors trained on photographs are picky about silhouettes, so the head,
    shoulders, swinging arms and legs all have to read at a glance — a plain
    rectangle gets ignored.
    """
    h = height
    x, y = int(cx), int(cy)
    top = y - h // 2
    head_r = max(3, int(h * 0.085))
    shoulder = int(h * 0.20)
    hip = int(h * 0.55)
    half_torso = max(3, int(h * 0.115))
    limb = max(2, int(h * 0.055))
    swing = np.sin(phase)

    # head and neck
    cv2.circle(frame, (x, top + head_r), head_r, shade, -1, cv2.LINE_AA)
    cv2.line(frame, (x, top + head_r), (x, top + shoulder), shade, limb, cv2.LINE_AA)

    # arms swing opposite the legs, and sit outside the torso so the gap shows
    for sign in (-1, 1):
        cv2.line(
            frame,
            (x + sign * half_torso, top + shoulder + limb),
            (x + sign * int(half_torso * 1.7), top + hip - int(h * 0.04) + int(swing * sign * h * 0.05)),
            shade, limb, cv2.LINE_AA,
        )

    # tapered torso
    torso = np.array([
        [x - half_torso, top + shoulder],
        [x + half_torso, top + shoulder],
        [x + int(half_torso * 0.82), top + hip],
        [x - int(half_torso * 0.82), top + hip],
    ], dtype=np.int32)
    cv2.fillConvexPoly(frame, torso, shade, cv2.LINE_AA)

    # legs
    for sign in (-1, 1):
        knee_x = x + sign * int(half_torso * 0.5)
        foot_x = knee_x + int(swing * sign * h * 0.12)
        cv2.line(frame, (x + sign * int(half_torso * 0.45), top + hip),
                 (knee_x, top + int(h * 0.78)), shade, limb, cv2.LINE_AA)
        cv2.line(frame, (knee_x, top + int(h * 0.78)), (foot_x, top + h), shade, limb, cv2.LINE_AA)


def _background() -> np.ndarray:
    bg = np.full((HEIGHT, WIDTH, 3), (140, 142, 145), dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (WIDTH, int(HEIGHT * 0.42)), (118, 120, 124), -1)   # wall
    cv2.rectangle(bg, (int(WIDTH * 0.34), int(HEIGHT * 0.06)),
                  (int(WIDTH * 0.66), int(HEIGHT * 0.42)), (96, 108, 120), -1)     # doorway
    for x in range(0, WIDTH, 96):                                                  # floor tiles
        cv2.line(bg, (x, int(HEIGHT * 0.42)), (x, HEIGHT), (128, 130, 133), 1)
    return bg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="samples/sample_doorway.mp4")
    ap.add_argument("--enters", type=int, default=3, help="people walking in (upward)")
    ap.add_argument("--exits", type=int, default=2, help="people walking out (downward)")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--lead-in", type=float, default=2.0,
                    help="seconds of empty scene before the first person walks")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(args.seconds * FPS)
    count = args.enters + args.exits
    # An empty lead-in: motion detectors need a few seconds of empty scene to
    # learn the background, and every detector needs frames on the near side of
    # the line before a crossing can be recognised.
    lead_in = int(args.lead_in * FPS)
    # Walk one person at a time down their own lane. Overlapping synthetic
    # figures merge into a single blob that any detector reads as one person,
    # which would make the ground truth a lie rather than a test.
    slot = max(60, (total_frames - lead_in) // max(1, count))
    people = []
    for i in range(count):
        entering = i < args.enters
        span = min(slot - 12, rng.randint(60, 90))
        start_f = lead_in + i * slot + rng.randint(0, max(1, slot - span - 1))
        lane = (i + 0.5) / count
        people.append(
            dict(
                entering=entering,
                start=start_f,
                end=min(total_frames, start_f + span),
                x=WIDTH * (0.22 + 0.56 * lane),
                drift=rng.uniform(-30, 30),
                height=rng.randint(170, 200),
                shade=tuple(int(c) for c in rng.sample(range(28, 90), 3)),
            )
        )

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    bg = _background()

    y_top, y_bottom = HEIGHT * 0.26, HEIGHT * 0.86
    for f in range(total_frames):
        frame = bg.copy()
        for p in people:
            if not (p["start"] <= f <= p["end"]):
                continue
            t = (f - p["start"]) / max(1, p["end"] - p["start"])
            cy = (y_bottom + (y_top - y_bottom) * t) if p["entering"] else (y_top + (y_bottom - y_top) * t)
            cx = p["x"] + p["drift"] * t
            scale = 0.85 + 0.3 * (cy / HEIGHT)   # perspective: nearer == bigger
            _draw_person(frame, cx, cy, int(p["height"] * scale), t * 14.0, p["shade"])
        writer.write(frame)
    writer.release()

    truth = {"video": str(out), "expected_entered": args.enters, "expected_exited": args.exits,
             "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": total_frames,
             "lead_in_seconds": args.lead_in}
    Path(out.with_suffix(".truth.json")).write_text(json.dumps(truth, indent=2))
    print(f"wrote {out} ({total_frames} frames) — expect {args.enters} in / {args.exits} out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
