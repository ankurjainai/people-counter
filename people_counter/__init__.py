"""Count people entering and exiting a shop or room in a video clip."""

from .config import CountingConfig, CountingLine
from .counter import Counts, CrossingEvent, LineCounter
from .detectors import DETECTORS, Detection, build_detector
from .pipeline import CountResult, count_people, grab_frame, probe
from .tracker import CentroidTracker, Track

__version__ = "1.0.0"

__all__ = [
    "CountingConfig",
    "CountingLine",
    "Counts",
    "CrossingEvent",
    "LineCounter",
    "DETECTORS",
    "Detection",
    "build_detector",
    "CountResult",
    "count_people",
    "grab_frame",
    "probe",
    "CentroidTracker",
    "Track",
    "__version__",
]
