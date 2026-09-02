"""Geometry, tracking and counting — no detector, so results are deterministic."""

import unittest

from people_counter import CountingConfig, CountingLine, Detection
from people_counter.counter import LineCounter
from people_counter.tracker import CentroidTracker

W, H = 640, 480


def walk(counter, tracker, xs_ys, start_frame=0, fps=25.0):
    """Feed a single moving detection through tracker + counter."""
    for i, (x, y) in enumerate(xs_ys):
        det = Detection(x - 20, y - 40, x + 20, y + 40, 0.9)
        tracks = tracker.update([det])
        counter.update(tracks, start_frame + i, (start_frame + i) / fps)


class LineGeometryTests(unittest.TestCase):
    def test_horizontal_line_inside_is_above(self):
        line = CountingLine.horizontal(0.5)
        self.assertGreater(line.side(320, 100, W, H), 0, "above the line is inside")
        self.assertLess(line.side(320, 400, W, H), 0, "below the line is outside")

    def test_pixels_scale_with_frame(self):
        line = CountingLine(0.0, 0.25, 1.0, 0.75)
        self.assertEqual(line.pixels(W, H), (0, 120, 640, 360))

    def test_rejects_out_of_range_and_degenerate(self):
        with self.assertRaises(ValueError):
            CountingLine(0, 0, 1, 1.5)
        with self.assertRaises(ValueError):
            CountingLine(0.5, 0.5, 0.5, 0.5)

    def test_roundtrip_dict(self):
        line = CountingLine(0.1, 0.2, 0.3, 0.4, inside_is_left=False)
        self.assertEqual(CountingLine.from_dict(line.to_dict()), line)


class CountingTests(unittest.TestCase):
    def setUp(self):
        self.config = CountingConfig(line=CountingLine.horizontal(0.5), min_hits=2)
        self.tracker = CentroidTracker(frame_size=(W, H))
        self.counter = LineCounter(self.config, W, H)

    def test_walking_up_counts_as_entry(self):
        walk(self.counter, self.tracker, [(320, y) for y in range(400, 100, -20)])
        self.assertEqual((self.counter.counts.entered, self.counter.counts.exited), (1, 0))

    def test_walking_down_counts_as_exit(self):
        walk(self.counter, self.tracker, [(320, y) for y in range(100, 400, 20)])
        self.assertEqual((self.counter.counts.entered, self.counter.counts.exited), (0, 1))

    def test_flipped_line_reverses_direction(self):
        config = CountingConfig(line=CountingLine.horizontal(0.5, inside_is_left=False), min_hits=2)
        counter = LineCounter(config, W, H)
        walk(counter, CentroidTracker(frame_size=(W, H)), [(320, y) for y in range(400, 100, -20)])
        self.assertEqual((counter.counts.entered, counter.counts.exited), (0, 1))

    def test_person_never_crossing_is_not_counted(self):
        walk(self.counter, self.tracker, [(320, y) for y in range(400, 300, -10)])
        self.assertEqual((self.counter.counts.entered, self.counter.counts.exited), (0, 0))

    def test_each_person_counts_once_even_if_they_linger(self):
        path = [(320, y) for y in range(400, 200, -20)] + [(320, y) for y in range(200, 260, 20)]
        walk(self.counter, self.tracker, path)
        self.assertEqual(self.counter.counts.entered, 1)
        self.assertEqual(len(self.counter.counts.events), 1)

    def test_min_hits_suppresses_a_one_frame_flicker(self):
        config = CountingConfig(line=CountingLine.horizontal(0.5), min_hits=5)
        counter = LineCounter(config, W, H)
        walk(counter, CentroidTracker(frame_size=(W, H)), [(320, 300), (320, 200)])
        self.assertEqual((counter.counts.entered, counter.counts.exited), (0, 0))

    def test_two_people_crossing_opposite_ways(self):
        tracker, counter = self.tracker, self.counter
        for i in range(14):
            up = Detection(80, 400 - i * 25, 120, 480 - i * 25, 0.9)
            down = Detection(500, 60 + i * 25, 540, 140 + i * 25, 0.9)
            tracks = tracker.update([up, down])
            counter.update(tracks, i, i / 25.0)
        self.assertEqual((counter.counts.entered, counter.counts.exited), (1, 1))

    def test_events_carry_timestamps_and_directions(self):
        walk(self.counter, self.tracker, [(320, y) for y in range(400, 100, -20)], fps=25.0)
        event = self.counter.counts.events[0]
        self.assertEqual(event.direction, "in")
        self.assertGreater(event.timestamp, 0.0)
        self.assertIn("track_id", event.to_dict())

    def test_vertical_line_counts_horizontal_movement(self):
        config = CountingConfig(line=CountingLine(0.5, 0.0, 0.5, 1.0), min_hits=2)
        counter = LineCounter(config, W, H)
        walk(counter, CentroidTracker(frame_size=(W, H)), [(x, 240) for x in range(500, 150, -25)])
        # The segment points down the frame, so its left half-plane is the left
        # half of the image: moving right-to-left crosses inward.
        self.assertEqual((counter.counts.entered, counter.counts.exited), (1, 0))


class TrackerTests(unittest.TestCase):
    def test_identity_is_kept_across_frames(self):
        tracker = CentroidTracker(frame_size=(W, H))
        for i in range(6):
            tracker.update([Detection(100 + i * 8, 100, 140 + i * 8, 180, 0.9)])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(next(iter(tracker.tracks.values())).hits, 6)

    def test_a_teleporting_detection_starts_a_new_track(self):
        tracker = CentroidTracker(max_distance=0.05, frame_size=(W, H))
        tracker.update([Detection(0, 0, 40, 80, 0.9)])
        tracker.update([Detection(600, 400, 640, 480, 0.9)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_track_survives_a_brief_occlusion(self):
        tracker = CentroidTracker(max_age=5, frame_size=(W, H))
        tracker.update([Detection(100, 100, 140, 180, 0.9)])
        for _ in range(3):
            tracker.update([])
        tracker.update([Detection(104, 104, 144, 184, 0.9)])
        self.assertEqual(len(tracker.tracks), 1)

    def test_track_is_dropped_after_max_age(self):
        tracker = CentroidTracker(max_age=3, frame_size=(W, H))
        tracker.update([Detection(100, 100, 140, 180, 0.9)])
        for _ in range(5):
            tracker.update([])
        self.assertEqual(len(tracker.tracks), 0)

    def test_two_people_keep_separate_ids(self):
        tracker = CentroidTracker(frame_size=(W, H))
        for i in range(5):
            tracker.update([
                Detection(100 + i * 5, 100, 140 + i * 5, 180, 0.9),
                Detection(400 - i * 5, 300, 440 - i * 5, 380, 0.9),
            ])
        self.assertEqual(len(tracker.tracks), 2)


if __name__ == "__main__":
    unittest.main()
