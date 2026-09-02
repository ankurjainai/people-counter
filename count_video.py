#!/usr/bin/env python3
"""Command-line entry point: count people crossing a line in a video."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from people_counter import DETECTORS, CountingConfig, CountingLine, count_people, probe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="count_video.py",
        description="Count people entering and exiting across a virtual line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="path to the video clip")
    p.add_argument("-o", "--output", help="write an annotated MP4 here")
    p.add_argument("--json", dest="json_out", help="write the full result as JSON here")

    line = p.add_argument_group("counting line (normalised 0..1 coordinates)")
    line.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                      help="line endpoints; default is a horizontal line at --line-y")
    line.add_argument("--line-y", type=float, default=0.5,
                      help="height of the default horizontal line")
    line.add_argument("--flip", action="store_true",
                      help="swap which side counts as inside")

    det = p.add_argument_group("detector")
    det.add_argument("--detector", choices=list(DETECTORS), default="auto",
                     help="auto picks YOLO when ultralytics is installed, else motion")
    det.add_argument("--model", default="yolov8n.pt", help="ultralytics weights")
    det.add_argument("--confidence", type=float, default=0.35)
    det.add_argument("--imgsz", type=int, default=640)
    det.add_argument("--device", help="cpu | mps | cuda:0 (default: auto)")

    run = p.add_argument_group("run")
    run.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    run.add_argument("--max-frames", type=int, help="stop after N processed frames")
    run.add_argument("--min-hits", type=int, default=3,
                     help="detections before a track may be counted")
    run.add_argument("--max-age", type=int, default=30,
                     help="frames a track survives without a detection")
    run.add_argument("--no-trails", action="store_true", help="do not draw motion trails")
    run.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    video = Path(args.video)
    if not video.exists():
        print(f"error: no such file: {video}", file=sys.stderr)
        return 2

    line = (
        CountingLine(*args.line, inside_is_left=not args.flip)
        if args.line
        else CountingLine.horizontal(args.line_y, inside_is_left=not args.flip)
    )
    config = CountingConfig(
        line=line,
        detector=args.detector,
        model=args.model,
        confidence=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        frame_stride=args.stride,
        max_frames=args.max_frames,
        min_hits=args.min_hits,
        max_age=args.max_age,
        write_video=bool(args.output),
        draw_trails=not args.no_trails,
    )

    info = probe(video)
    if not args.quiet:
        print(f"{video.name}: {info.width}x{info.height} @ {info.fps:.1f} fps, "
              f"{info.frame_count or '?'} frames")

    def show(p) -> None:
        if args.quiet:
            return
        pct = f"{p.fraction * 100:5.1f}%" if p.fraction is not None else f"{p.frames_done:6d}f"
        print(f"\r  {pct}  in={p.entered} out={p.exited}  {p.elapsed:5.1f}s", end="", flush=True)

    result = count_people(video, config, output_path=args.output, progress=show)
    if not args.quiet:
        print()

    print(result.summary())
    for event in result.counts.events:
        print(f"  t={event.timestamp:7.2f}s  track #{event.track_id:<3d} {event.direction.upper()}")
    if result.output_path:
        print(f"Annotated video: {result.output_path}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
