"""HTTP-level tests for the Flask front-end."""

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import app as webapp

from .support import write_blank_video


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        # Keep the test's uploads and outputs out of the real project folders.
        cls._orig = (webapp.UPLOAD_DIR, webapp.OUTPUT_DIR)
        webapp.UPLOAD_DIR = tmp / "uploads"
        webapp.OUTPUT_DIR = tmp / "outputs"
        cls.source = write_blank_video(tmp / "clip.mp4")
        cls.app = webapp.create_app()
        cls.app.config["TESTING"] = True

    @classmethod
    def tearDownClass(cls):
        webapp.UPLOAD_DIR, webapp.OUTPUT_DIR = cls._orig
        cls._tmp.cleanup()

    def setUp(self):
        self.client = self.app.test_client()

    def upload(self, name="clip.mp4"):
        data = {"video": (io.BytesIO(self.source.read_bytes()), name)}
        return self.client.post("/api/upload", data=data, content_type="multipart/form-data")

    # -- pages -----------------------------------------------------------
    def test_index_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"People Counter", res.data)

    # -- upload ----------------------------------------------------------
    def test_upload_returns_video_metadata(self):
        res = self.upload()
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["video_id"])
        self.assertEqual(body["width"], 320)
        self.assertEqual(body["height"], 240)

    def test_upload_without_a_file_is_rejected(self):
        res = self.client.post("/api/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("No video file", res.get_json()["error"])

    def test_upload_rejects_a_non_video_extension(self):
        data = {"video": (io.BytesIO(b"not a video"), "notes.txt")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file type", res.get_json()["error"])

    def test_upload_rejects_an_undecodable_file(self):
        data = {"video": (io.BytesIO(b"\x00\x01 not really an mp4"), "broken.mp4")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

    # -- preview ---------------------------------------------------------
    def test_preview_returns_a_jpeg(self):
        video_id = self.upload().get_json()["video_id"]
        res = self.client.get(f"/api/preview/{video_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "image/jpeg")
        self.assertTrue(res.data.startswith(b"\xff\xd8"))

    def test_preview_of_an_unknown_id_is_404(self):
        self.assertEqual(self.client.get("/api/preview/deadbeef00").status_code, 404)

    def test_preview_rejects_a_path_traversal_id(self):
        self.assertEqual(self.client.get("/api/preview/..%2f..%2fetc").status_code, 404)

    # -- jobs ------------------------------------------------------------
    def test_job_requires_a_known_video(self):
        res = self.client.post("/api/jobs", json={"video_id": "nosuchid1234"})
        self.assertEqual(res.status_code, 404)

    def test_job_rejects_an_unknown_detector(self):
        video_id = self.upload().get_json()["video_id"]
        res = self.client.post("/api/jobs", json={"video_id": video_id, "detector": "magic"})
        self.assertEqual(res.status_code, 400)

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/api/jobs/doesnotexist").status_code, 404)

    def test_job_runs_to_completion_and_exposes_results(self):
        video_id = self.upload().get_json()["video_id"]
        res = self.client.post("/api/jobs", json={
            "video_id": video_id,
            "detector": "motion",        # no weights to download
            "max_frames": 3,
            "line": {"y": 0.5},
            "write_video": True,
        })
        self.assertEqual(res.status_code, 202)
        job_id = res.get_json()["job_id"]

        job = self._await_job(job_id)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["result"]["frames_processed"], 3)
        self.assertEqual(job["result"]["detector"], "motion")

        with self.client.get(f"/api/jobs/{job_id}/video") as video:
            self.assertEqual(video.status_code, 200)

        with self.client.get(f"/api/jobs/{job_id}/result.json") as payload:
            self.assertEqual(payload.status_code, 200)
            self.assertIn("entered", json.loads(payload.data))

        listing = self.client.get("/api/jobs").get_json()["jobs"]
        self.assertIn(job_id, [j["job_id"] for j in listing])

    def test_cancelling_a_finished_job_is_a_conflict(self):
        video_id = self.upload().get_json()["video_id"]
        job_id = self.client.post("/api/jobs", json={
            "video_id": video_id, "detector": "motion", "max_frames": 1, "write_video": False,
        }).get_json()["job_id"]
        self._await_job(job_id)
        res = self.client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(res.status_code, 409)

    def _await_job(self, job_id, timeout=90.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").get_json()
            if job["status"] in {"done", "failed", "cancelled"}:
                return job
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish within {timeout}s")


if __name__ == "__main__":
    unittest.main()
