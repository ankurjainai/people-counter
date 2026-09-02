"""Background job manager for the web front-end.

Each upload becomes a :class:`Job` processed on a worker thread, so the browser
can poll for progress instead of holding a request open for minutes. State lives
in memory plus the files on disk — this is a single-process tool, not a cluster.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from people_counter import CountingConfig, build_detector, count_people
from people_counter.pipeline import Progress

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"


@dataclass
class Job:
    job_id: str
    video_path: Path
    original_name: str
    config: CountingConfig
    output_path: Path
    status: str = QUEUED
    stage: str = "Queued"
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    frames_done: int = 0
    frames_total: Optional[int] = None
    entered: int = 0
    exited: int = 0
    error: Optional[str] = None
    result: Optional[dict] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def fraction(self) -> Optional[float]:
        if not self.frames_total:
            return None
        return min(1.0, self.frames_done / self.frames_total)

    def to_dict(self) -> dict:
        elapsed = (self.finished or time.time()) - (self.started or self.created)
        eta = None
        frac = self.fraction
        if self.status == RUNNING and frac and frac > 0.02:
            eta = round(elapsed / frac - elapsed, 1)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "original_name": self.original_name,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "progress": round(frac, 4) if frac is not None else None,
            "entered": self.entered,
            "exited": self.exited,
            "occupancy": self.entered - self.exited,
            "elapsed": round(elapsed, 1),
            "eta": eta,
            "error": self.error,
            "result": self.result,
            "has_video": bool(self.result and self.result.get("output_path")),
        }


class JobManager:
    """Runs one job at a time; extra submissions queue behind it."""

    def __init__(self, output_dir: Path, max_jobs: int = 50) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_jobs = max_jobs
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()   # serialises the actual decoding

    def submit(self, video_path: Path, original_name: str, config: CountingConfig) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            video_path=Path(video_path),
            original_name=original_name,
            config=config,
            output_path=self.output_dir / f"{job_id}.mp4",
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._evict_locked()
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"job-{job_id}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[dict]:
        with self._lock:
            jobs = [self._jobs[j] for j in reversed(self._order) if j in self._jobs]
        return [j.to_dict() for j in jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status in (DONE, FAILED, CANCELLED):
            return False
        job._cancel.set()
        return True

    # -- internals -------------------------------------------------------
    def _evict_locked(self) -> None:
        while len(self._order) > self.max_jobs:
            old_id = self._order.pop(0)
            old = self._jobs.pop(old_id, None)
            if old is None:
                continue
            for path in (old.output_path, old.video_path):
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best effort cleanup
                    log.debug("could not remove %s", path)

    def _run(self, job: Job) -> None:
        with self._worker_lock:
            if job._cancel.is_set():
                job.status = CANCELLED
                job.stage = "Cancelled"
                job.finished = time.time()
                return
            job.status = RUNNING
            job.started = time.time()

            def on_progress(p: Progress) -> None:
                job.stage = "Counting"
                job.frames_done = p.frames_done
                job.frames_total = p.frames_total
                job.entered = p.entered
                job.exited = p.exited

            try:
                # Built here, not inside count_people, so the browser can show
                # "Loading detector" while a first-run model download happens.
                job.stage = "Loading detector"
                detector = build_detector(job.config)
                job.stage = "Counting"
                result = count_people(
                    job.video_path,
                    job.config,
                    detector=detector,
                    output_path=job.output_path if job.config.write_video else None,
                    progress=on_progress,
                    progress_every=5,
                    cancelled=job._cancel.is_set,
                )
                job.result = result.to_dict()
                job.entered = result.counts.entered
                job.exited = result.counts.exited
                job.status = CANCELLED if job._cancel.is_set() else DONE
                job.stage = "Cancelled" if job._cancel.is_set() else "Done"
                (self.output_dir / f"{job.job_id}.json").write_text(
                    json.dumps(job.result, indent=2)
                )
            except Exception as exc:  # surfaced to the browser, not swallowed
                log.exception("job %s failed", job.job_id)
                job.status = FAILED
                job.stage = "Failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
