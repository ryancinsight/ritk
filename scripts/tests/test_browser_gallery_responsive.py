"""Responsive RITK gallery contract tests."""
from __future__ import annotations

import pathlib
import sys
import unittest

_scripts_root = pathlib.Path(__file__).resolve().parents[1]
_metis_root = pathlib.Path(__file__).resolve().parents[3] / "metis"
sys.path.insert(0, str(_scripts_root))
sys.path.insert(0, str(_metis_root / "scripts"))

import browser_gallery


class ResponsiveGalleryTests(unittest.TestCase):
    def test_declares_direct_container_and_hosted_matrix_entry(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        gallery_root = root / "crates" / "ritk-snap" / "web" / "gallery"
        html = (gallery_root / "gallery.html").read_text(encoding="utf-8")
        css = (gallery_root / "gallery.css").read_text(encoding="utf-8")
        script = (gallery_root / "gallery.js").read_text(encoding="utf-8")
        artifacts = (root / "scripts" / "browser_gallery_artifacts.py").read_text(encoding="utf-8")
        runner = (root / "scripts" / "browser_gallery.py").read_text(encoding="utf-8")
        workflow = (root / ".github" / "workflows" / "metis-browser-dicom.yml").read_text(encoding="utf-8")
        self.assertIn('id="responsive-views"', html)
        self.assertIn('query.get("layout") === "responsive"', script)
        self.assertIn("start_web_responsive_canvases", script)
        self.assertIn("data-ritk-pane-layout", script)
        self.assertIn("responsive-crosshair-overlay", script)
        self.assertIn("responsive-crosshair-overlay", runner)
        self.assertIn(".responsive-views", css)
        self.assertIn("position: relative", css)
        self.assertIn("responsive-views", artifacts)
        self.assertIn("result: chromium-responsive", workflow)
        self.assertIn("result: firefox-responsive", workflow)
        self.assertIn("canvas_capture: screenshot", workflow)
        self.assertIn("layout: responsive", workflow)
        self.assertIn('--page-query "layout=$LAYOUT"', workflow)
        self.assertIn("--crosshair-controls", workflow)
        self.assertIn("display-only projection", workflow)
        self.assertIn("generic canvas trace requires every selected canvas", workflow)

    def test_capture_sample_preserves_quad_roles_and_listeners(self):
        class Client:
            def execute(self, script):
                self.script = script
                return {
                    "responsive_layout": True,
                    "pane_layout": "quad",
                    "pane_roles": ["axial", "coronal", "sagittal", "projection"],
                    "consumer_listeners": 3,
                    "projection_mode": "mip",
                    "projection_statistic": "MIP",
                }

        sample = browser_gallery._responsive_capture_sample(Client())
        self.assertEqual(
            sample,
            {
                "layout": "quad",
                "roles": ["axial", "coronal", "sagittal", "projection"],
                "consumer_listeners": 3,
                "projection_mode": "mip",
                "projection_statistic": "MIP",
            },
        )
