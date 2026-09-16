"""Trusted keyboard and pointer actions for the browser gallery."""

from __future__ import annotations

from typing import Any, Mapping

from browser_canvas import settle_canvas_input
from browser_protocol import WebDriverClient
from browser_gallery_trace import (
    ARROW_BATCH_SIZE,
    _consume_events,
    _snapshot,
    _validate_transition,
)


def _keyboard_action(
    client: WebDriverClient,
    expected_counts: Mapping[str, int],
    actions: list[dict[str, Any]],
    axis: str,
    key: str,
    label: str,
    before: dict[str, dict[str, Any]],
    expected_index: int,
) -> dict[str, dict[str, Any]]:
    """Dispatch one native key and retain exact state and trusted-event evidence."""
    client.key_press(key, source_id=f"metis-gallery-{axis}-keyboard")
    settle_canvas_input(client)
    after = _snapshot(client, expected_counts)
    changed = before[axis]["index"] != after[axis]["index"]
    required = ["keydown", "keyup"] + (["input", "change"] if changed else [])
    events = _consume_events(client, axis, required, expected_key=key)
    _validate_transition(before, after, axis, expected_index)
    actions.append(
        {
            "axis": axis,
            "action": label,
            "from_index": before[axis]["index"],
            "to_index": after[axis]["index"],
            "generation": after[axis]["generation"],
            "rgba_sha256": after[axis]["rgba_sha256"],
            "events": events,
        }
    )
    return after


def _arrow_batch(
    client: WebDriverClient,
    expected_counts: Mapping[str, int],
    actions: list[dict[str, Any]],
    axis: str,
    before: dict[str, dict[str, Any]],
    presses: int,
) -> dict[str, dict[str, Any]]:
    """Restore a slice with one bounded W3C ArrowRight action batch."""
    key_value = "\ue014"
    source_actions = []
    for _ in range(presses):
        source_actions.extend(
            ({"type": "keyDown", "value": key_value}, {"type": "keyUp", "value": key_value})
        )
    client.perform_actions(
        [
            {
                "type": "key",
                "id": f"metis-gallery-{axis}-restore",
                "actions": source_actions,
            }
        ]
    )
    settle_canvas_input(client)
    after = _snapshot(client, expected_counts)
    expected_index = before[axis]["index"] + presses
    events = _consume_events(
        client,
        axis,
        ("keydown", "keyup", "input", "change"),
        expected_key="ArrowRight",
    )
    _validate_transition(before, after, axis, expected_index)
    actions.append(
        {
            "axis": axis,
            "action": "restore-arrow-right-batch",
            "presses": presses,
            "from_index": before[axis]["index"],
            "to_index": after[axis]["index"],
            "generation": after[axis]["generation"],
            "rgba_sha256": after[axis]["rgba_sha256"],
            "events": events,
        }
    )
    return after
