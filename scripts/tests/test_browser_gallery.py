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
