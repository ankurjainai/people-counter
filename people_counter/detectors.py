"""Person detectors.

Every backend returns the same thing: a list of axis-aligned ``Detection``
boxes in pixel coordinates for one BGR frame.

* :class:`YoloDetector`   - ultralytics YOLO, class 0 ("person"). Accurate, and
  the right choice whenever it can be installed.
* :class:`MotionDetector` - MOG2 background subtraction plus person-shaped blob
  filtering. No weights, no torch. Assumes a fixed camera, and cannot tell a
  person from anything else person-sized that moves.
* :class:`HogDetector`    - OpenCV's classic HOG + linear SVM pedestrian
  detector. Available only on OpenCV 4.x; OpenCV 5 dropped it.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Downloaded YOLO weights are cached here rather than in whatever directory the
# server happened to start in.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_LOCK = threading.Lock()


def resolve_weights(model: str) -> str:
    """Return a path to ``model``, preferring the project's models/ cache.

    A bare asset name ("yolov8n.pt") that is not cached yet is handed to
    ultralytics, which downloads it into the working directory; the file is then
    moved into models/ so the next run — and any other working directory — finds
    it locally.
    """
    given = Path(model)
    if given.is_file():
        return str(given)
    cached = MODEL_DIR / given.name
    if cached.is_file():
        return str(cached)
    return model                      # let ultralytics fetch it by name


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_ints(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class Detector:
    """Interface implemented by every backend."""

    name = "base"

    def detect(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover
        raise NotImplementedError


class YoloDetector(Detector):
    name = "yolo"

    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.model_name = model

        with _MODEL_LOCK:
            weights = resolve_weights(model)
            self._model = YOLO(weights)
            if weights == model and not Path(model).is_file():
                self._cache_downloaded_weights(Path(model).name)

    @staticmethod
    def _cache_downloaded_weights(name: str) -> None:
        """Move weights ultralytics just downloaded into models/ for reuse."""
        downloaded = Path.cwd() / name
        if not downloaded.is_file():
            return
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(MODEL_DIR / name))
            log.info("cached %s in %s", name, MODEL_DIR)
        except OSError as exc:  # pragma: no cover - cosmetic caching only
            log.debug("could not cache %s: %s", name, exc)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        kwargs = dict(
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=[0],          # COCO class 0 == person
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self._model.predict(frame, **kwargs)
        out: List[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(xyxy, conf):
                out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(score)))
        return out


class HogDetector(Detector):
    """Fallback detector — OpenCV's pedestrian HOG descriptor."""

    name = "hog"

    def __init__(self, confidence: float = 0.35, target_width: int = 640) -> None:
        import cv2

        if not hasattr(cv2, "HOGDescriptor"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no HOGDescriptor (removed in OpenCV 5). "
                "Use the 'motion' detector, or install opencv-python-headless<5."
            )
        self._cv2 = cv2
        self.confidence = confidence
        self.target_width = target_width
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        scale = 1.0
        work = frame
        if w > self.target_width:
            scale = self.target_width / float(w)
            work = cv2.resize(frame, (self.target_width, int(round(h * scale))))

        rects, weights = self._hog.detectMultiScale(
            work, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        out: List[Detection] = []
        # HOG weights are SVM margins, not probabilities. Map the usual useful
        # range (0..1.5) onto 0..1 so one `confidence` knob works for both
        # backends.
        for (x, y, bw, bh), weight in zip(rects, np.asarray(weights).reshape(-1)):
            score = float(min(1.0, max(0.0, weight / 1.5)))
            if score < self.confidence:
                continue
            out.append(
                Detection(
                    x / scale, y / scale, (x + bw) / scale, (y + bh) / scale, score
                )
            )
        return _non_max_suppression(out, iou_threshold=0.45)


class MotionDetector(Detector):
    """Background subtraction — the dependency-free fallback.

    Learns what the empty scene looks like, then keeps the moving blobs whose
    size and aspect ratio are plausible for a standing person. It needs a fixed
    camera and a few seconds of footage to settle, and it will happily count a
    moving trolley as a person; YOLO is better whenever it is available.
    """

    name = "motion"

    def __init__(
        self,
        confidence: float = 0.35,
        target_width: int = 640,
        history: int = 300,
        min_area_fraction: float = 0.004,
        max_area_fraction: float = 0.45,
        aspect_range: tuple[float, float] = (0.15, 1.4),
        warmup: int = 12,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.target_width = target_width
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.aspect_range = aspect_range
        self.warmup = warmup
        self._frames = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=28, detectShadows=True
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        scale = 1.0
        work = frame
        if w > self.target_width:
            scale = self.target_width / float(w)
            work = cv2.resize(frame, (self.target_width, int(round(h * scale))))

        mask = self._subtractor.apply(work)
        self._frames += 1
        # Shadows come back as 127; only hard foreground (255) is a person.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        if self._frames <= self.warmup:
            return []                      # the model has not settled yet

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = work.shape[0] * work.shape[1]
        out: List[Detection] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            fraction = area / frame_area
            if not (self.min_area_fraction <= fraction <= self.max_area_fraction):
                continue
            aspect = bw / bh if bh else 99.0
            if not (self.aspect_range[0] <= aspect <= self.aspect_range[1]):
                continue
            fill = cv2.contourArea(contour) / area if area else 0.0
            if fill < 0.30:                # a thin, spidery blob is not a body
                continue
            out.append(
                Detection(x / scale, y / scale, (x + bw) / scale, (y + bh) / scale, float(fill))
            )
        return _non_max_suppression(out, iou_threshold=0.4)


def _non_max_suppression(dets: Sequence[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    kept: List[Detection] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


DETECTORS = ("auto", "yolo", "motion", "hog")


def build_detector(config) -> Detector:
    """Instantiate the detector named by ``config`` (a :class:`CountingConfig`)."""
    kind = config.resolved_detector()
    if kind == "yolo":
        try:
            return YoloDetector(
                model=config.model,
                confidence=config.confidence,
                imgsz=config.imgsz,
                device=config.device,
            )
        except Exception as exc:
            if config.detector == "yolo":
                raise
            log.warning("YOLO unavailable (%s); falling back to motion detection", exc)
            kind = "motion"
    if kind == "motion":
        return MotionDetector(confidence=config.confidence, target_width=config.imgsz)
    if kind == "hog":
        return HogDetector(confidence=config.confidence, target_width=config.imgsz)
    raise ValueError(f"Unknown detector: {kind!r}. Choose one of {', '.join(DETECTORS)}.")
