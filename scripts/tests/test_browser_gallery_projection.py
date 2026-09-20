"""Projection-specific checks for the RITK browser gallery harness."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_metis_root = pathlib.Path(__file__).resolve().parents[3] / "metis"
_ritk_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(_metis_root / "scripts"))
import browser_gallery
import browser_gallery_projection
import browser_canvas


class ProjectionGalleryTests(unittest.TestCase):
    def test_capture_records_real_pixels_and_listener_budget(self):
        class Client:
            def execute_async(self, _script, _arguments):
                return {
                    "width": 512,
                    "height": 512,
                    "non_black_pixels": 131_072,
                    "rgba_sha256": "a" * 64,
                }

            def execute(self, script, _arguments=()):
                if "window.metisGallery.sample" in script:
                    return {
                        "projection_mode": "mip",
                        "projection_statistic": "MIP",
                        "consumer_listeners": 21,
                    }
                return {
                    "attributes": {
                        "data-ritk-role": "projection",
                        "data-ritk-load-state": "ready",
                        "data-ritk-frame-state": "presented",
                        "data-ritk-projection-statistic": "MIP",
                        "data-ritk-frame-width": "512",
                        "data-ritk-frame-height": "512",
                    }
                }

        oracle = {
            "ritk-snap-projection": {
                "width": 512,
                "height": 512,
                "attributes": {
                    "data-ritk-role": "projection",
                    "data-ritk-load-state": "ready",
                    "data-ritk-frame-state": "presented",
                    "data-ritk-projection-statistic": "MIP",
                    "data-ritk-frame-width": "512",
                    "data-ritk-frame-height": "512",
                },
            }
        }
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-projection-"
        ) as directory:
            evidence = browser_gallery_projection.capture_projection_gallery(
                Client(), pathlib.Path(directory) / "projection", oracle, statistic="mip"
            )
            self.assertEqual(evidence["projection"]["non_black_pixels"], 131_072)
            self.assertTrue(evidence["projection"]["display_only"])
            self.assertEqual(
                json.loads(
                    (pathlib.Path(directory) / "projection" / "projection.json").read_text(
                        encoding="utf-8"
                    )
                ),
                evidence["projection"],
            )

    def test_consumer_callback_accepts_four_canvas_projection_mode(self):
        class Client:
            def set_window_rect(self, width, height):
                self.rect = (width, height)

        oracle = {
            f"ritk-snap-{axis}": {
                "attributes": {"data-ritk-slice-count": str(count)},
            }
            for axis, count in zip(browser_gallery.AXES, (94, 512, 512))
        }
        oracle["ritk-snap-projection"] = {"width": 512, "height": 512, "attributes": {}}
        canvas_ids = tuple(
            [f"ritk-snap-{axis}" for axis in browser_gallery.AXES]
            + ["ritk-snap-projection"]
        )
        client = Client()
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-projection-callback-"
        ) as directory, mock.patch.object(
            browser_gallery,
            "capture_slice_gallery",
            return_value={"schema": 1, "kind": "slices"},
        ) as slices, mock.patch.object(
            browser_gallery,
            "capture_projection_gallery",
            return_value={"projection": {"schema": 1}},
        ) as projection:
            result = browser_gallery._capture_consumer_controls(
                client,
                pathlib.Path(directory),
                oracle,
                canvas_ids,
                projection="mip",
            )

        self.assertEqual(result["projection"], {"schema": 1})
        slices.assert_called_once_with(
            client,
            pathlib.Path(directory) / "slices",
            expected_counts={"axial": 94, "coronal": 512, "sagittal": 512},
        )
        projection.assert_called_once_with(
            client,
            pathlib.Path(directory) / "projection",
            oracle,
            statistic="mip",
        )

    def test_host_trace_attribute_budget_is_preserved(self):
        workflow = (
            _ritk_root / ".github" / "workflows" / "metis-browser-dicom.yml"
        ).read_text(encoding="utf-8")
        self.assertLessEqual(
            workflow.count("--canvas-attribute"), browser_canvas.MAX_CANVAS_ATTRIBUTES
        )
