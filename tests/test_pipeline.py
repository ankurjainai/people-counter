"""End-to-end pipeline tests driven by a scripted detector (no model needed)."""

import json
import tempfile
import unittest
from pathlib import Path

from people_counter import (
    CountingConfig,
    CountingLine,
    build_detector,
    count_people,
    grab_frame,
    probe,
)

from .support import FPS, FRAMES, HEIGHT, WIDTH, ScriptedDetector, straight_walk, write_blank_video


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.video = write_blank_video(cls.tmp / "clip.mp4")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_probe_reads_dimensions(self):
        info = probe(self.video)
        self.assertEqual((info.width, info.height), (WIDTH, HEIGHT))
        self.assertAlmostEqual(info.fps, FPS, delta=1.0)
        self.assertEqual(info.frame_count, FRAMES)

    def test_probe_rejects_a_missing_file(self):
        with self.assertRaises(ValueError):
            probe(self.tmp / "nope.mp4")

    def test_grab_frame_returns_a_frame(self):
        frame = grab_frame(self.video, 0)
        self.assertEqual(frame.shape[:2], (HEIGHT, WIDTH))

    def test_counts_one_entry_for_an_upward_walk(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(line=CountingLine.horizontal(0.5), min_hits=2, write_video=False)
        result = count_people(self.video, config, detector=detector)
        self.assertEqual((result.counts.entered, result.counts.exited), (1, 0))
        self.assertEqual(result.frames_processed, FRAMES)

    def test_counts_one_exit_for_a_downward_walk(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.1, HEIGHT * 0.9, FRAMES))
        config = CountingConfig(line=CountingLine.horizontal(0.5), min_hits=2, write_video=False)
        result = count_people(self.video, config, detector=detector)
        self.assertEqual((result.counts.entered, result.counts.exited), (0, 1))

    def test_writes_a_playable_annotated_video(self):
        out = self.tmp / "annotated.mp4"
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(line=CountingLine.horizontal(0.5), min_hits=2)
        result = count_people(self.video, config, output_path=out, detector=detector)
        self.assertTrue(out.exists() and out.stat().st_size > 0)
        self.assertEqual(result.output_path, str(out))
        self.assertEqual(probe(out).frame_count, FRAMES)

    def test_max_frames_stops_early(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(max_frames=5, write_video=False)
        result = count_people(self.video, config, detector=detector)
        self.assertEqual(result.frames_processed, 5)

    def test_stride_processes_fewer_frames(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(frame_stride=4, write_video=False)
        result = count_people(self.video, config, detector=detector)
        self.assertEqual(result.frames_processed, (FRAMES + 3) // 4)

    def test_cancellation_stops_the_run(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(write_video=False)
        result = count_people(self.video, config, detector=detector, cancelled=lambda: True)
        self.assertEqual(result.frames_processed, 1)

    def test_progress_callback_is_invoked(self):
        seen = []
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(write_video=False)
        count_people(self.video, config, detector=detector, progress=seen.append, progress_every=5)
        self.assertTrue(seen)
        self.assertEqual(seen[-1].frames_done, FRAMES)
        self.assertAlmostEqual(seen[-1].fraction, 1.0, places=3)

    def test_result_serialises_to_json(self):
        detector = ScriptedDetector(straight_walk(160, HEIGHT * 0.9, HEIGHT * 0.1, FRAMES))
        config = CountingConfig(min_hits=2, write_video=False)
        payload = count_people(self.video, config, detector=detector).to_dict()
        json.dumps(payload)                     # must not raise
        self.assertEqual(payload["entered"], 1)
        self.assertEqual(payload["occupancy"], 1)
        self.assertEqual(payload["events"][0]["direction"], "in")


class DetectorSelectionTests(unittest.TestCase):
    def test_auto_resolves_to_an_available_backend(self):
        config = CountingConfig(detector="auto")
        self.assertIn(config.resolved_detector(), {"yolo", "motion"})

    def test_explicit_choice_is_respected(self):
        self.assertEqual(CountingConfig(detector="motion").resolved_detector(), "motion")

    def test_unknown_detector_is_rejected(self):
        with self.assertRaises(ValueError):
            build_detector(CountingConfig(detector="telepathy"))

    def test_motion_detector_finds_a_moving_blob(self):
        import numpy as np
        from people_counter.detectors import MotionDetector

        detector = MotionDetector(warmup=5, target_width=320)
        background = np.full((240, 320, 3), 30, dtype=np.uint8)
        for _ in range(12):
            detector.detect(background.copy())
        frame = background.copy()
        frame[60:190, 140:190] = 240              # a tall person-shaped blob
        boxes = detector.detect(frame)
        self.assertTrue(boxes, "a moving person-sized blob should be detected")


if __name__ == "__main__":
    unittest.main()
