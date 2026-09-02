#!/usr/bin/env python3
"""Flask front-end: upload a clip, draw the counting line, watch the counts."""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path

import cv2
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from jobs import JobManager
from people_counter import (
    DETECTORS,
    CountingConfig,
    CountingLine,
    __version__,
    grab_frame,
    probe,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
MAX_UPLOAD_MB = int(os.environ.get("PC_MAX_UPLOAD_MB", "512"))

log = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = JobManager(OUTPUT_DIR)
    app.extensions["job_manager"] = manager

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template(
            "index.html", version=__version__, max_upload_mb=MAX_UPLOAD_MB
        )

    # ----------------------------------------------------------------- API
    @app.post("/api/upload")
    def upload():
        file = request.files.get("video")
        if file is None or not file.filename:
            return jsonify(error="No video file in the request."), 400

        name = secure_filename(file.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXT:
            return (
                jsonify(error=f"Unsupported file type {suffix or '(none)'}. "
                              f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"),
                400,
            )

        video_id = uuid.uuid4().hex[:12]
        path = UPLOAD_DIR / f"{video_id}{suffix}"
        file.save(path)

        try:
            info = probe(path)
        except ValueError as exc:
            path.unlink(missing_ok=True)
            return jsonify(error=f"Could not read that video: {exc}"), 400
        if info.width <= 0 or info.height <= 0:
            path.unlink(missing_ok=True)
            return jsonify(error="That file has no decodable video stream."), 400

        return jsonify(video_id=video_id, original_name=name, **info.to_dict())

    @app.get("/api/preview/<video_id>")
    def preview(video_id: str):
        """A JPEG still the browser draws the counting line on top of."""
        path = _find_upload(video_id)
        if path is None:
            abort(404)
        index = request.args.get("frame", default=0, type=int)
        try:
            frame = grab_frame(path, max(0, index))
        except ValueError:
            abort(404)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            abort(500)
        return app.response_class(buf.tobytes(), mimetype="image/jpeg")

    @app.post("/api/jobs")
    def create_job():
        data = request.get_json(silent=True) or {}
        video_id = data.get("video_id")
        path = _find_upload(video_id) if video_id else None
        if path is None:
            return jsonify(error="Unknown video_id — upload the clip again."), 404

        try:
            config = _config_from_payload(data)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=f"Bad settings: {exc}"), 400

        job = manager.submit(path, data.get("original_name") or path.name, config)
        return jsonify(job.to_dict()), 202

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify(jobs=manager.list())

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.to_dict())

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        if not manager.cancel(job_id):
            return jsonify(error="Job is not running."), 409
        return jsonify(ok=True)

    @app.get("/api/jobs/<job_id>/video")
    def job_video(job_id: str):
        job = manager.get(job_id)
        if job is None or not job.output_path.exists():
            abort(404)
        return send_file(job.output_path, mimetype="video/mp4", conditional=True)

    @app.get("/api/jobs/<job_id>/result.json")
    def job_json(job_id: str):
        job = manager.get(job_id)
        if job is None or job.result is None:
            abort(404)
        return send_from_directory(
            OUTPUT_DIR, f"{job_id}.json", as_attachment=True,
            download_name=f"count-{job_id}.json",
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error=f"That file is larger than the {MAX_UPLOAD_MB} MB limit."), 413

    return app


def _find_upload(video_id: str) -> Path | None:
    """Resolve an upload id to a path, refusing anything that escapes the dir."""
    if not video_id or not video_id.isalnum():
        return None
    for candidate in UPLOAD_DIR.glob(f"{video_id}.*"):
        if candidate.is_file():
            return candidate
    return None


def _config_from_payload(data: dict) -> CountingConfig:
    line_data = data.get("line") or {}
    line = (
        CountingLine.from_dict(line_data)
        if {"x1", "y1", "x2", "y2"} <= set(line_data)
        else CountingLine.horizontal(
            float(line_data.get("y", 0.5)),
            inside_is_left=bool(line_data.get("inside_is_left", True)),
        )
    )
    detector = str(data.get("detector", "auto"))
    if detector not in set(DETECTORS):
        raise ValueError(f"unknown detector {detector!r}")

    return CountingConfig(
        line=line,
        detector=detector,
        model=str(data.get("model") or CountingConfig.model),
        confidence=_clamp(float(data.get("confidence", 0.35)), 0.01, 0.99),
        imgsz=int(data.get("imgsz", 640)),
        frame_stride=max(1, int(data.get("frame_stride", 1))),
        max_frames=int(data["max_frames"]) if data.get("max_frames") else None,
        min_hits=max(1, int(data.get("min_hits", 3))),
        max_age=max(1, int(data.get("max_age", 30))),
        write_video=bool(data.get("write_video", True)),
        draw_trails=bool(data.get("draw_trails", True)),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the people-counter web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"\n  People Counter — http://{args.host}:{args.port}\n")
    # threaded=True so status polling stays responsive while a job decodes.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


# Module-level app so `flask run` and WSGI servers (gunicorn app:app) work too.
app = create_app()

if __name__ == "__main__":
    raise SystemExit(main())
