"""RITK consumer-owned saved-study gallery harness tests."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_metis_root = pathlib.Path(__file__).resolve().parents[3] / "metis"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(_metis_root / "scripts"))
import browser_gallery
import browser_gallery_cine
import browser_gallery_tools
import browser_gallery_window


class SliceGalleryTests(unittest.TestCase):
    def test_slice_gallery_runner_restores_state_through_split_helpers(self):
        counts = {axis: 32 for axis in browser_gallery.AXES}

        class Client:
            def __init__(self):
                self.states = {
                    axis: {
                        "index": 17,
                        "generation": 1,
                        "rgba_sha256": f"{18:064x}",
                        "slider_width": 100.0,
                    }
                    for axis in browser_gallery.AXES
                }
                self.pointer_drags = []
                self.release_count = 0

            def set_index(self, axis, index):
                state = self.states[axis]
                if state["index"] != index:
                    state["index"] = index
                    state["generation"] += 1
                state["rgba_sha256"] = f"{index + 1:064x}"

            def snapshot(self):
                return {
                    axis: dict(state)
                    for axis, state in self.states.items()
                }

            def find(self, selector):
                return selector

            def click(self, element):
                axis = element.removeprefix("#slice-")
                self.set_index(axis, self.states[axis]["index"] + 1)

            def pointer_drag(self, element, start, end, *, source_id):
                del source_id
                axis = element.removeprefix("#slice-")
                self.pointer_drags.append((axis, start, end))
                self.set_index(axis, 0)

            def release_actions(self):
                self.release_count += 1

        client = Client()
        settle_calls = []

        def snapshot(target, expected_counts):
            self.assertEqual(expected_counts, counts)
            return target.snapshot()

        def keyboard_action(target, expected_counts, actions, axis, key, label, before, expected_index):
            self.assertEqual(expected_counts, counts)
            target.set_index(axis, expected_index)
            after = target.snapshot()
            actions.append(
                {
                    "axis": axis,
                    "action": label,
                    "from_index": before[axis]["index"],
                    "to_index": after[axis]["index"],
                    "generation": after[axis]["generation"],
                    "rgba_sha256": after[axis]["rgba_sha256"],
                    "events": {"event_count": 2},
                }
            )
            return after

        def arrow_batch(target, expected_counts, actions, axis, before, presses):
            self.assertEqual(expected_counts, counts)
            target.set_index(axis, before[axis]["index"] + presses)
            after = target.snapshot()
            actions.append(
                {
                    "axis": axis,
                    "action": "restore-arrow-right-batch",
                    "presses": presses,
                    "from_index": before[axis]["index"],
                    "to_index": after[axis]["index"],
                    "generation": after[axis]["generation"],
                    "rgba_sha256": after[axis]["rgba_sha256"],
                    "events": {"event_count": presses * 2},
                }
            )
            return after

        with (
            mock.patch.object(browser_gallery, "_snapshot", side_effect=snapshot),
            mock.patch.object(browser_gallery, "_probe_invalid_slice_api", return_value=[]),
            mock.patch.object(browser_gallery, "_install_event_trace", return_value=24),
            mock.patch.object(browser_gallery, "_cleanup_event_trace", return_value=24),
            mock.patch.object(browser_gallery, "_consume_events", return_value={"event_count": 1}),
            mock.patch.object(browser_gallery, "_validate_transition"),
            mock.patch.object(browser_gallery, "_keyboard_action", side_effect=keyboard_action),
            mock.patch.object(browser_gallery, "_arrow_batch", side_effect=arrow_batch),
            mock.patch.object(browser_gallery, "_write_gallery_screenshots", return_value={"window": {}}),
            mock.patch.object(
                browser_gallery,
                "settle_canvas_input",
                side_effect=lambda _client: settle_calls.append(True),
            ),
        ):
            with tempfile.TemporaryDirectory(
                dir=browser_gallery.ROOT / "output", prefix="gallery-import-"
            ) as directory:
                evidence = browser_gallery.capture_slice_gallery(
                    client,
                    pathlib.Path(directory),
                    expected_counts=counts,
                )
                evidence_path = pathlib.Path(directory) / "gallery-slices.json"
                self.assertEqual(json.loads(evidence_path.read_text(encoding="utf-8")), evidence)

        self.assertEqual(client.release_count, 1)
        self.assertEqual(len(settle_calls), len(browser_gallery.AXES) * 2)
        self.assertEqual(
            client.pointer_drags,
            [
                (axis, (47, 0), (-47, 0))
                for axis in browser_gallery.AXES
            ],
        )
        self.assertEqual(len(evidence["actions"]), len(browser_gallery.AXES) * 7)
        self.assertEqual(
            [
                action["presses"]
                for action in evidence["actions"]
                if action["action"] == "restore-arrow-right-batch"
            ],
            [16, 1] * len(browser_gallery.AXES),
        )
        for axis in browser_gallery.AXES:
            self.assertEqual(evidence["initial"][axis]["index"], 17)
            self.assertEqual(evidence["restored"][axis]["index"], 17)
            self.assertEqual(
                evidence["restored"][axis]["rgba_sha256"],
                evidence["initial"][axis]["rgba_sha256"],
            )

    def test_consumer_callback_maps_opaque_oracle_to_slice_counts(self):
        class Client:
            def __init__(self):
                self.rect = None

            def set_window_rect(self, width, height):
                self.rect = (width, height)

        oracle = {
            f"ritk-snap-{axis}": {
                "attributes": {"data-ritk-slice-count": str(count)},
            }
            for axis, count in zip(browser_gallery.AXES, (94, 512, 512))
        }
        canvas_ids = tuple(f"ritk-snap-{axis}" for axis in browser_gallery.AXES)
        client = Client()
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-callback-"
        ) as directory, mock.patch.object(
            browser_gallery,
            "capture_slice_gallery",
            return_value={"schema": 1},
        ) as capture:
            result = browser_gallery._capture_consumer_controls(
                client, pathlib.Path(directory), oracle, canvas_ids
            )

        self.assertEqual(result, {"schema": 1})
        self.assertEqual(client.rect, (1440, 1200))
        capture.assert_called_once_with(
            client,
            pathlib.Path(directory) / "slices",
            expected_counts={"axial": 94, "coronal": 512, "sagittal": 512},
        )

    def test_consumer_callback_can_capture_window_presets_after_slices(self):
        class Client:
            def set_window_rect(self, width, height):
                self.rect = (width, height)

        oracle = {
            f"ritk-snap-{axis}": {
                "attributes": {"data-ritk-slice-count": str(count)},
            }
            for axis, count in zip(browser_gallery.AXES, (94, 512, 512))
        }
        canvas_ids = tuple(f"ritk-snap-{axis}" for axis in browser_gallery.AXES)
        client = Client()
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-window-callback-"
        ) as directory, mock.patch.object(
            browser_gallery,
            "capture_slice_gallery",
            return_value={"schema": 1, "kind": "slices"},
        ) as slices, mock.patch.object(
            browser_gallery,
            "capture_window_preset_gallery",
            return_value={"schema": 1, "kind": "window-level"},
        ) as presets:
            result = browser_gallery._capture_consumer_controls(
                client,
                pathlib.Path(directory),
                oracle,
                canvas_ids,
                window_presets=True,
            )

        self.assertEqual(
            result,
            {
                "slices": {"schema": 1, "kind": "slices"},
                "window_level": {"schema": 1, "kind": "window-level"},
            },
        )
        slices.assert_called_once_with(
            client,
            pathlib.Path(directory) / "slices",
            expected_counts={"axial": 94, "coronal": 512, "sagittal": 512},
        )
        presets.assert_called_once_with(client, pathlib.Path(directory) / "window-level")

    def test_consumer_callback_can_capture_cine_controls_after_slices(self):
        class Client:
            def set_window_rect(self, width, height):
                self.rect = (width, height)

        oracle = {
            f"ritk-snap-{axis}": {
                "attributes": {"data-ritk-slice-count": str(count)},
            }
            for axis, count in zip(browser_gallery.AXES, (94, 512, 512))
        }
        canvas_ids = tuple(f"ritk-snap-{axis}" for axis in browser_gallery.AXES)
        client = Client()
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-cine-callback-"
        ) as directory, mock.patch.object(
            browser_gallery,
            "capture_slice_gallery",
            return_value={"schema": 1, "kind": "slices"},
        ) as slices, mock.patch.object(
            browser_gallery,
            "capture_cine_gallery",
            return_value={"schema": 1, "kind": "cine"},
        ) as cine:
            result = browser_gallery._capture_consumer_controls(
                client,
                pathlib.Path(directory),
                oracle,
                canvas_ids,
                cine_controls=True,
            )

        self.assertEqual(
            result,
            {
                "slices": {"schema": 1, "kind": "slices"},
                "cine": {"schema": 1, "kind": "cine"},
            },
        )
        slices.assert_called_once_with(
            client,
            pathlib.Path(directory) / "slices",
            expected_counts={"axial": 94, "coronal": 512, "sagittal": 512},
        )
        cine.assert_called_once_with(client, pathlib.Path(directory) / "cine")

    def test_consumer_callback_can_capture_tool_controls_after_slices(self):
        class Client:
            def set_window_rect(self, width, height):
                self.rect = (width, height)

        oracle = {
            f"ritk-snap-{axis}": {
                "attributes": {"data-ritk-slice-count": str(count)},
            }
            for axis, count in zip(browser_gallery.AXES, (94, 512, 512))
        }
        canvas_ids = tuple(f"ritk-snap-{axis}" for axis in browser_gallery.AXES)
        client = Client()
        with tempfile.TemporaryDirectory(
            dir=browser_gallery.ROOT / "output", prefix="gallery-tools-callback-"
        ) as directory, mock.patch.object(
            browser_gallery,
            "capture_slice_gallery",
            return_value={"schema": 1, "kind": "slices"},
        ) as slices, mock.patch.object(
            browser_gallery,
            "capture_tool_gallery",
            return_value={"schema": 1, "kind": "tools"},
        ) as tools:
            result = browser_gallery._capture_consumer_controls(
                client,
                pathlib.Path(directory),
                oracle,
                canvas_ids,
                tool_controls=True,
            )

        self.assertEqual(
            result,
            {
                "slices": {"schema": 1, "kind": "slices"},
                "tools": {"schema": 1, "kind": "tools"},
            },
        )
        slices.assert_called_once_with(
            client,
            pathlib.Path(directory) / "slices",
            expected_counts={"axial": 94, "coronal": 512, "sagittal": 512},
        )
        tools.assert_called_once_with(client, pathlib.Path(directory) / "tools")


class WindowPresetHelperTests(unittest.TestCase):
    def test_gallery_declares_rust_owned_window_preset_surface(self):
        gallery_root = pathlib.Path(__file__).resolve().parents[2] / "crates" / "ritk-snap" / "web" / "gallery"
        html = (gallery_root / "gallery.html").read_text(encoding="utf-8")
        script = (gallery_root / "gallery.js").read_text(encoding="utf-8")
        self.assertIn('id="window-preset"', html)
        self.assertIn('id="window-level"', html)
        self.assertIn("set_web_window_preset", script)
        self.assertIn("web_window_preset_count", script)
        self.assertIn("web_window_preset_name", script)
        self.assertIn("const syncAll = () => {", script)
        self.assertIn("      syncPresentation();", script)

    def test_gallery_declares_cine_controls_and_typed_api(self):
        gallery_root = pathlib.Path(__file__).resolve().parents[2] / "crates" / "ritk-snap" / "web" / "gallery"
        html = (gallery_root / "gallery.html").read_text(encoding="utf-8")
        script = (gallery_root / "gallery.js").read_text(encoding="utf-8")
        self.assertIn('id="cine-toggle"', html)
        self.assertIn('id="cine-rate"', html)
        self.assertIn('id="cine-rate-value"', html)
        self.assertIn("toggle_web_cine", script)
        self.assertIn("set_web_cine_rate", script)
        self.assertIn("data-ritk-cine-enabled", script)
        self.assertIn("canvases.every", script)

    def test_gallery_declares_diagnostic_tool_palette_and_typed_api(self):
        gallery_root = pathlib.Path(__file__).resolve().parents[2] / "crates" / "ritk-snap" / "web" / "gallery"
        html = (gallery_root / "gallery.html").read_text(encoding="utf-8")
        script = (gallery_root / "gallery.js").read_text(encoding="utf-8")
        self.assertIn('id="tool-buttons"', html)
        self.assertIn('id="active-tool"', html)
        self.assertIn("select_web_tool", script)
        self.assertIn("web_tool_count", script)
        self.assertIn("web_tool_name", script)
        self.assertIn("data-ritk-active-tool-index", script)
        self.assertIn("data-ritk-active-tool", script)
        self.assertIn("canvases.every", script)

    def test_tool_probe_contract_rejects_invalid_indices(self):
        for value in ("NaN", "Infinity", "-Infinity", "-1", "0.5", "4294967296"):
            self.assertIn(value, browser_gallery_tools.INVALID_TOOL_API_PROBE_SCRIPT)

    def test_cine_rate_parser_rejects_non_integer_and_out_of_range_values(self):
        self.assertEqual(browser_gallery_cine.MAX_CINE_RATE, 60)
        for value in ("0", "0.5", "61", "4294967296"):
            self.assertIn(value, browser_gallery_cine.INVALID_CINE_API_PROBE_SCRIPT)

    def test_cine_snapshot_uses_the_synchronous_webdriver_contract(self):
        script = browser_gallery_cine.CINE_SNAPSHOT_SCRIPT
        self.assertNotIn("arguments[arguments.length - 1]", script)
        self.assertIn("return {ok: true", script)

    def test_cine_frame_wait_compares_snapshot_slice_field_and_preserves_timeout_state(self):
        script = browser_gallery_cine.WAIT_CINE_FRAME_SCRIPT
        self.assertIn("slice_index: rawIndex === null ? null : Number(rawIndex)", script)
        self.assertIn("state.slice_index !== previous[position].slice_index", script)
        self.assertIn("previous[position].frame_generation", script)
        self.assertNotIn("previous[position].generation", script)
        self.assertNotIn("state.index !== previous[position].index", script)
        self.assertIn("done({ok: false, previous, current: read(), status: status()})", script)

    def test_decimal_parser_rejects_non_decimal_or_unbounded_values(self):
        with self.assertRaises(browser_gallery_window.BrowserRuntimeError):
            browser_gallery_window._decimal("1.0", "index")
        with self.assertRaises(browser_gallery_window.BrowserRuntimeError):
            browser_gallery_window._decimal(str(1 << 54), "index")
