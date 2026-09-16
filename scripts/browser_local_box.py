"""Verify RITK canvas input through styled browser content boxes.

The script reuses Metis's bounded WebDriver transport and generated gallery,
but owns every RITK-specific assertion.  It loads the same study in fresh
unstyled and styled sessions, clicks the same backing-frame voxel centre, and
compares the linked cursor reconstructed from the three orthogonal slice
attributes.  It also checks wheel input in the content and padding regions.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
from typing import Any


CANVAS_IDS = (
    "ritk-snap-axial",
    "ritk-snap-coronal",
    "ritk-snap-sagittal",
)
TARGET_CANVAS = CANVAS_IDS[0]
# The unstyled gallery's integer W3C lattice skips row 341.  The pre-dispatch
# rounding oracle proves this nearby voxel is representable in every layout.
TARGET_FRACTION = ((173.5 / 512.0), (340.5 / 512.0))
WHEEL_DELTA = 120

from browser_local_box_page import (
    BORDER_BOX_CSS,
    FRAME_GENERATION_SCRIPT,
    GEOMETRY_SCRIPT,
    INSTALL_EVENTS_SCRIPT,
    READ_EVENTS_SCRIPT,
    SLICE_VECTOR_SCRIPT,
    STYLED_CSS,
    WAIT_FOR_FRAME_SCRIPT,
)
from browser_local_box_provenance import bundle_manifest, copy_bundle, repository_state


def _rounding_oracle(geometry: dict[str, Any]) -> dict[str, Any]:
    content = geometry["content"]
    delta = [
        content["client"]["x"] - content["exact"]["x"],
        content["client"]["y"] - content["exact"]["y"],
    ]
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    for transform in geometry["transform_chain"]:
        outer_a = transform["a"] * a + transform["c"] * b
        outer_b = transform["b"] * a + transform["d"] * b
        outer_c = transform["a"] * c + transform["c"] * d
        outer_d = transform["b"] * c + transform["d"] * d
        a, b, c, d = outer_a, outer_b, outer_c, outer_d
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0.0:
        raise RuntimeError(f"browser transform is not invertible: {determinant!r}")
    local_delta = [
        (d * delta[0] - c * delta[1]) / determinant,
        (-b * delta[0] + a * delta[1]) / determinant,
    ]
    nominal = [
        geometry["frame_width"] * TARGET_FRACTION[0],
        geometry["frame_height"] * TARGET_FRACTION[1],
    ]
    mapped = [
        nominal[0] + local_delta[0] / geometry["content_width"] * geometry["frame_width"],
        nominal[1] + local_delta[1] / geometry["content_height"] * geometry["frame_height"],
    ]
    return {
        "viewport_rounding_delta": delta,
        "inverse_local_delta": local_delta,
        "mapped_frame": mapped,
        "predicted_voxel": [math.floor(mapped[1]), math.floor(mapped[0])],
    }


def _load_metis(metis_root: pathlib.Path) -> dict[str, Any]:
    scripts = metis_root / "scripts"
    if not (scripts / "browser_protocol.py").is_file():
        raise RuntimeError(f"Metis browser scripts were not found under {scripts}")
    sys.path.insert(0, str(scripts))
    from browser_canvas import settle_canvas_input
    from browser_drop import study_files
    from browser_protocol import StaticServer, WebDriverClient, parse_device_scale
    from browser_runtime import _wait_for_selector, _wait_for_text

    return {
        "settle": settle_canvas_input,
        "study_files": study_files,
        "StaticServer": StaticServer,
        "WebDriverClient": WebDriverClient,
        "parse_device_scale": parse_device_scale,
        "wait_selector": _wait_for_selector,
        "wait_text": _wait_for_text,
    }


def _slice_vector(client: Any) -> list[int]:
    values = client.execute(SLICE_VECTOR_SCRIPT, [list(CANVAS_IDS)])
    if not isinstance(values, list) or len(values) != len(CANVAS_IDS):
        raise RuntimeError(f"RITK slice vector is malformed: {values!r}")
    vector = []
    for value in values:
        if not isinstance(value, dict):
            raise RuntimeError(f"RITK slice state is malformed: {value!r}")
        index = value.get("index")
        count = value.get("count")
        if (
            not isinstance(index, int)
            or not isinstance(count, int)
            or count <= 0
            or not 0 <= index < count
        ):
            raise RuntimeError(f"RITK slice state is outside its bounds: {value!r}")
        vector.append(index)
    return vector


def _dispatch_click(client: Any, point: dict[str, int], source_id: str) -> None:
    client.perform_actions(
        [
            {
                "type": "pointer",
                "id": source_id,
                "parameters": {"pointerType": "mouse"},
                "actions": [
                    {"type": "pointerMove", "origin": "viewport", **point, "duration": 0},
                    {"type": "pointerDown", "button": 0},
                    {"type": "pointerUp", "button": 0},
                ],
            }
        ]
    )


def _dispatch_wheel(client: Any, point: dict[str, int], source_id: str) -> None:
    client.perform_actions(
        [
            {
                "type": "wheel",
                "id": source_id,
                "actions": [
                    {
                        "type": "scroll",
                        "origin": "viewport",
                        **point,
                        "deltaX": 0,
                        "deltaY": WHEEL_DELTA,
                        "duration": 0,
                    }
                ],
            }
        ]
    )


def _trusted_events(client: Any, expected: tuple[str, ...], point: dict[str, int]) -> list[dict[str, Any]]:
    events = client.execute(READ_EVENTS_SCRIPT)
    if not isinstance(events, list):
        raise RuntimeError("browser event evidence is unavailable")
    observed = tuple(event.get("type") for event in events)
    if observed != expected:
        raise RuntimeError(f"browser emitted {observed!r}; expected {expected!r}")
    for event in events:
        if (
            event.get("is_trusted") is not True
            or event.get("target_id") != TARGET_CANVAS
            or event.get("client_x") != point["x"]
            or event.get("client_y") != point["y"]
        ):
            raise RuntimeError(f"browser event lacks trusted target evidence: {event!r}")
        if event.get("type") == "wheel" and event.get("delta_y") != WHEEL_DELTA:
            raise RuntimeError(f"browser wheel event has the wrong delta: {event!r}")
    return events


def _wait_for_new_frame(client: Any, previous: int) -> int:
    result = client.execute_async(WAIT_FOR_FRAME_SCRIPT, [TARGET_CANVAS, previous])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"RITK did not present a new frame after input: {result!r}")
    generation = result.get("generation")
    if not isinstance(generation, int):
        raise RuntimeError(f"RITK returned an invalid frame generation: {generation!r}")
    return generation


def _case(
    modules: dict[str, Any],
    bundle: pathlib.Path,
    files: list[pathlib.Path],
    args: argparse.Namespace,
    layout: str,
) -> dict[str, Any]:
    client = modules["WebDriverClient"](args.driver_url, 120)
    loaded_artifacts = bundle_manifest(bundle)
    actions_released = False
    try:
        with modules["StaticServer"](bundle) as origin:
            client.create_session(
                args.browser_name,
                args.device_scale_milli,
                headless=args.headless,
            )
            client.set_timeouts(120_000)
            client._request(
                "POST",
                client._session_path("window/rect"),
                {"width": 1800, "height": 1200},
            )
            client.navigate(origin + "gallery.html")
            modules["wait_text"](client, "gallery-status", "Ready.", timeout_ms=30_000, include=True)
            client.send_file_paths(client.find("#file-input"), files)
            for canvas_id in CANVAS_IDS:
                modules["wait_selector"](
                    client,
                    f'#{canvas_id}[data-ritk-load-state="ready"]'
                    '[data-ritk-frame-state="presented"]',
                    timeout_ms=120_000,
                )
            if client.execute(INSTALL_EVENTS_SCRIPT, [TARGET_CANVAS]) is not True:
                raise RuntimeError("browser input observer could not be installed")
            geometry = client.execute(GEOMETRY_SCRIPT, [TARGET_CANVAS, list(TARGET_FRACTION)])
            if not isinstance(geometry, dict) or geometry.get("error") is not None:
                raise RuntimeError(f"browser geometry is unavailable: {geometry!r}")
            content = geometry.get("content")
            if not isinstance(content, dict) or content.get("target_id") != TARGET_CANVAS:
                raise RuntimeError(f"content target does not hit the RITK canvas: {content!r}")
            point = content.get("client")
            if not isinstance(point, dict) or not all(isinstance(point.get(key), int) for key in ("x", "y")):
                raise RuntimeError(f"content target is not an integer W3C point: {point!r}")

            generation = client.execute(FRAME_GENERATION_SCRIPT, [TARGET_CANVAS])
            if not isinstance(generation, int) or generation <= 0:
                raise RuntimeError(f"RITK frame generation is invalid: {generation!r}")
            initial = _slice_vector(client)
            frame_width = geometry.get("frame_width")
            frame_height = geometry.get("frame_height")
            if (
                not isinstance(frame_width, int)
                or not isinstance(frame_height, int)
                or frame_width <= 0
                or frame_height <= 0
            ):
                raise RuntimeError("RITK frame dimensions are unavailable")
            expected_click = [
                initial[0],
                math.floor(frame_height * TARGET_FRACTION[1]),
                math.floor(frame_width * TARGET_FRACTION[0]),
            ]
            rounding_oracle = _rounding_oracle(geometry)
            if rounding_oracle["predicted_voxel"] != expected_click[1:]:
                raise RuntimeError(
                    f"integer W3C rounding predicts {rounding_oracle['predicted_voxel']!r}; "
                    f"nominal target is {expected_click[1:]!r}; layout={layout}"
                )
            _dispatch_click(client, point, "ritk-local-box-click")
            generation = _wait_for_new_frame(client, generation)
            click_events = _trusted_events(client, ("pointerdown", "pointerup"), point)
            clicked = _slice_vector(client)
            if clicked != expected_click:
                raise RuntimeError(
                    f"content click selected {clicked!r}; expected exact voxel {expected_click!r}; "
                    f"layout={layout}, target={content!r}"
                )

            padding_events = None
            border_events = None
            if layout != "unstyled":
                padding = geometry.get("padding")
                if not isinstance(padding, dict) or padding.get("target_id") != TARGET_CANVAS:
                    raise RuntimeError(f"padding target does not hit the RITK canvas: {padding!r}")
                padding_point = padding.get("client")
                if not isinstance(padding_point, dict):
                    raise RuntimeError("padding target has no W3C point")
                _dispatch_wheel(client, padding_point, "ritk-local-box-padding-wheel")
                modules["settle"](client)
                padding_events = _trusted_events(client, ("wheel",), padding_point)
                after_padding = _slice_vector(client)
                if after_padding != clicked:
                    raise RuntimeError(
                        "wheel input in canvas padding changed the RITK slice vector "
                        f"from {clicked!r} to {after_padding!r}"
                    )

                border = geometry.get("border")
                if not isinstance(border, dict) or border.get("target_id") != TARGET_CANVAS:
                    raise RuntimeError(f"border target does not hit the RITK canvas: {border!r}")
                border_point = border.get("client")
                if not isinstance(border_point, dict):
                    raise RuntimeError("border target has no W3C point")
                _dispatch_wheel(client, border_point, "ritk-local-box-border-wheel")
                modules["settle"](client)
                border_events = _trusted_events(client, ("wheel",), border_point)
                after_border = _slice_vector(client)
                if after_border != clicked:
                    raise RuntimeError(
                        "wheel input in canvas border changed the RITK slice vector "
                        f"from {clicked!r} to {after_border!r}"
                    )

            _dispatch_wheel(client, point, "ritk-local-box-content-wheel")
            _wait_for_new_frame(client, generation)
            wheel_events = _trusted_events(client, ("wheel",), point)
            after_wheel = _slice_vector(client)
            client.release_actions()
            actions_released = True
            return {
                "layout": layout,
                "bundle_artifacts": loaded_artifacts,
                "geometry": geometry,
                "rounding_oracle": rounding_oracle,
                "initial_voxel": initial,
                "expected_click_voxel": expected_click,
                "click_voxel": clicked,
                "after_wheel": after_wheel,
                "click_events": click_events,
                "padding_events": padding_events,
                "border_events": border_events,
                "wheel_events": wheel_events,
            }
    finally:
        if not actions_released and client.session_id is not None:
            client.release_actions()
        client.close()


def _validate_result(unstyled: dict[str, Any], styled_cases: list[dict[str, Any]]) -> None:
    for styled in styled_cases:
        if styled["click_voxel"] != unstyled["click_voxel"]:
            raise RuntimeError(
                f"{styled['layout']} click selected {styled['click_voxel']}; "
                f"unstyled selected {unstyled['click_voxel']}"
            )
        if styled["after_wheel"] != unstyled["after_wheel"]:
            raise RuntimeError(
                f"{styled['layout']} wheel selected {styled['after_wheel']}; "
                f"unstyled selected {unstyled['after_wheel']}"
            )
    delta = [
        after - before
        for before, after in zip(unstyled["click_voxel"], unstyled["after_wheel"])
    ]
    if delta != [-1, 0, 0]:
        raise RuntimeError(f"content wheel changed the wrong RITK voxel coordinate: {delta!r}")

    for styled in styled_cases:
        geometry = styled["geometry"]
        for name in ("border_left", "border_top", "padding_left", "padding_top"):
            value = geometry.get(name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise RuntimeError(f"{styled['layout']} has no measured {name}: {value!r}")
        for name in ("content_width", "content_height"):
            value = geometry.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value % 1.0 == 0.0:
                raise RuntimeError(f"{styled['layout']} lacks a fractional {name}: {value!r}")
        transform = geometry.get("transform")
        if not isinstance(transform, dict) or transform.get("b") == 0 or transform.get("c") == 0:
            raise RuntimeError(f"{styled['layout']} lacks ancestor rotation: {transform!r}")
        transform_chain = geometry.get("transform_chain")
        if not isinstance(transform_chain, list) or len(transform_chain) != 2:
            raise RuntimeError(f"{styled['layout']} lacks nested transforms: {transform_chain!r}")
        outer = transform_chain[1]
        if (not isinstance(outer, dict) or outer.get("a", 1.0) * outer.get("d", 1.0)
                - outer.get("b", 0.0) * outer.get("c", 0.0) >= 0.0):
            raise RuntimeError(f"{styled['layout']} lacks an ancestor reflection: {outer!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metis-root", required=True, type=pathlib.Path)
    parser.add_argument("--moirai-root", type=pathlib.Path)
    parser.add_argument("--driver-url", default="http://127.0.0.1:9515")
    parser.add_argument("--browser-name", default="MicrosoftEdge")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device-scale", default="1")
    parser.add_argument("--files", required=True, type=pathlib.Path)
    parser.add_argument("--pattern", default="*.dcm")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output/browser/local-box.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ritk_root = pathlib.Path(__file__).resolve().parents[1]
    args.metis_root = args.metis_root.resolve(strict=True)
    args.moirai_root = (
        args.moirai_root.resolve(strict=True)
        if args.moirai_root is not None
        else (args.metis_root.parent / "moirai").resolve(strict=True)
    )
    modules = _load_metis(args.metis_root)
    args.device_scale_milli = modules["parse_device_scale"](args.device_scale)
    files, _ = modules["study_files"](args.files.resolve(strict=True), args.pattern)
    source_bundle = args.metis_root / "output" / "browser"
    if not (source_bundle / "gallery.html").is_file():
        raise RuntimeError(f"built Metis gallery was not found under {source_bundle}")
    source_state = {
        "ritk": repository_state(ritk_root),
        "metis": repository_state(args.metis_root),
        "moirai": repository_state(args.moirai_root),
    }
    base_bundle_artifacts = bundle_manifest(source_bundle)

    output = (ritk_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output_root = (ritk_root / "output").resolve()
    if not output.is_relative_to(output_root):
        raise RuntimeError(f"output must stay under {output_root}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ritk-local-box-", dir=source_bundle) as temporary:
        root = pathlib.Path(temporary)
        unstyled_bundle = root / "unstyled"
        styled_bundle = root / "content-box"
        border_box_bundle = root / "border-box"
        copy_bundle(source_bundle, unstyled_bundle)
        copy_bundle(source_bundle, styled_bundle)
        copy_bundle(source_bundle, border_box_bundle)
        with (styled_bundle / "gallery.css").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n")
            stream.write(STYLED_CSS)
        with (border_box_bundle / "gallery.css").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n")
            stream.write(BORDER_BOX_CSS)
        unstyled = _case(modules, unstyled_bundle, files, args, "unstyled")
        content_box = _case(modules, styled_bundle, files, args, "content-box")
        border_box = _case(modules, border_box_bundle, files, args, "border-box")

    _validate_result(unstyled, [content_box, border_box])
    document = {
        "schema": 1,
        "status": "passed",
        "source_state": source_state,
        "base_bundle_artifacts": base_bundle_artifacts,
        "browser_name": args.browser_name,
        "device_scale": args.device_scale,
        "target_fraction": list(TARGET_FRACTION),
        "unstyled": unstyled,
        "content_box": content_box,
        "border_box": border_box,
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "click_voxel": unstyled["click_voxel"],
        "after_wheel": unstyled["after_wheel"],
        "output": str(output),
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
