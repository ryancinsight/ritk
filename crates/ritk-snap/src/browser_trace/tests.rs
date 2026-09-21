use super::*;
use serde_json::{json, Value};

mod cine_rate;
mod interaction;

// The standalone anisotropic study has physical [width, height] extents of
// [2, 3], [2, 6], and [3, 6] for axial, coronal, and sagittal slices.
const SAMPLE_PHYSICAL_EXTENTS: [[f64; 2]; 3] = [[2.0, 3.0], [2.0, 6.0], [3.0, 6.0]];
const CINE_CSS_HEIGHT: f64 = 153.2;

fn sample_display_aspect(axis: usize) -> f64 {
    let [width, height] = SAMPLE_PHYSICAL_EXTENTS[axis];
    width / height
}

pub(super) fn default_ids() -> Vec<String> {
    DEFAULT_CANVAS_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect()
}

fn fixture_value(ids: &[String]) -> Value {
    fixture_value_for_mode(ids, TraceInputMode::PointerWheel)
}

fn keyboard_fixture_value(ids: &[String]) -> Value {
    fixture_value_for_mode(
        ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
    )
}

pub(super) fn cine_rate_fixture_value(ids: &[String]) -> Value {
    fixture_value_for_mode(
        ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate),
    )
}

#[derive(Clone, Copy)]
struct SnapshotFixtureStage {
    suffix: &'static str,
    slice_index: usize,
    cine_state: Option<(u32, u64)>,
}

impl SnapshotFixtureStage {
    const fn new(suffix: &'static str, slice_index: usize, cine_state: Option<(u32, u64)>) -> Self {
        Self {
            suffix,
            slice_index,
            cine_state,
        }
    }
}

fn fixture_value_for_mode(ids: &[String], input_mode: TraceInputMode) -> Value {
    let attributes = |axis: usize, slice_index: usize, cine_state: Option<(u32, u64)>| {
        let mut value = json!({
            "data-ritk-load-state": "ready",
            "data-ritk-frame-state": "presented",
            "data-ritk-axis": axis.to_string(),
            "data-ritk-slice-index": slice_index.to_string(),
            "data-ritk-slice-count": "4",
            "data-ritk-frame-width": "256",
            "data-ritk-frame-height": "192",
            "data-ritk-crosshair-visible": "false",
            "data-ritk-linked-cursor": "128,96,128",
            "data-ritk-view-flip-h": "false",
            "data-ritk-view-flip-v": "false",
            "data-ritk-view-rotation": "0"
        });
        if let Some((rate, generation)) = cine_state {
            value["data-ritk-cine-fps"] = json!(rate.to_string());
            value["data-ritk-frame-generation"] = json!(generation.to_string());
            value["data-ritk-display-aspect"] = json!(sample_display_aspect(axis).to_string());
        }
        value
    };
    let cine_rate_mode = matches!(
        input_mode,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate)
    );
    let mut actions = Vec::new();
    for id in ids {
        match input_mode.keyboard_kind() {
            Some(KeyboardTraceKind::Navigation) => {
                actions.push(keyboard_action(id, "ArrowDown", "ArrowDown", false, true));
            }
            Some(KeyboardTraceKind::CineRate) => {
                actions.extend([
                    keyboard_action(id, "=", "Equal", false, false),
                    keyboard_action(id, "=", "Equal", true, true),
                    keyboard_action(id, "-", "Minus", false, false),
                    keyboard_action(id, "-", "Minus", true, true),
                ]);
            }
            None => {}
        }
        actions.push(json!({"action": "trusted-pointer-drag", "canvas": id}));
        actions.push(json!({"action": "trusted-wheel", "canvas": id}));
    }
    let snapshots: Vec<Value> = ids
        .iter()
        .enumerate()
        .flat_map(|(axis, id)| {
            let stages: Vec<SnapshotFixtureStage> = match input_mode {
                TraceInputMode::PointerWheel => vec![
                    SnapshotFixtureStage::new("initial", 0, None),
                    SnapshotFixtureStage::new("after-input", 1, None),
                ],
                TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation) => vec![
                    SnapshotFixtureStage::new("initial", 0, None),
                    SnapshotFixtureStage::new("after-keyboard", 1, None),
                    SnapshotFixtureStage::new("after-input", 2, None),
                ],
                TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate) => vec![
                    SnapshotFixtureStage::new("initial", 0, Some((12, 7))),
                    SnapshotFixtureStage::new("after-keyboard", 0, Some((13, 8))),
                    SnapshotFixtureStage::new("after-repeat", 0, Some((13, 8))),
                    SnapshotFixtureStage::new("after-decrease", 0, Some((12, 9))),
                    SnapshotFixtureStage::new("after-decrease-repeat", 0, Some((12, 9))),
                    SnapshotFixtureStage::new("after-input", 1, Some((12, 10))),
                ],
            };
            stages
                .into_iter()
                .map(|stage| {
                    let SnapshotFixtureStage {
                        suffix,
                        slice_index,
                        cine_state,
                    } = stage;
                    let (css_width, css_height) = if cine_rate_mode {
                        (
                            sample_display_aspect(axis) * CINE_CSS_HEIGHT,
                            CINE_CSS_HEIGHT,
                        )
                    } else {
                        (256.0, 192.0)
                    };
                    json!({
                        "label": format!("{id}-{suffix}"),
                        "canvas": {
                            "id": id,
                            "width": 256,
                            "height": 192,
                            "css_width": css_width,
                            "css_height": css_height,
                            "attributes": attributes(axis, slice_index, cine_state)
                        }
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let screenshot_suffixes = match input_mode {
        TraceInputMode::PointerWheel
        | TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation) => {
            ["initial", "after-input"].as_slice()
        }
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate) => {
            super::cine_rate::SNAPSHOT_SUFFIXES.as_slice()
        }
    };
    let screenshots: Vec<Value> = [
            json!({"label": "window-initial", "width": 1280, "height": 720, "bytes": 100, "sha256": "0".repeat(64)}),
            json!({"label": "window-final", "width": 1280, "height": 720, "bytes": 100, "sha256": "1".repeat(64)}),
        ]
        .into_iter()
        .chain(ids.iter().enumerate().flat_map(|(axis, id)| {
            screenshot_suffixes.iter().map(move |suffix| {
                let digest_digit = match *suffix {
                    "initial" => "2",
                    "after-keyboard" | "after-repeat" => "4",
                    "after-decrease" | "after-decrease-repeat" => "5",
                    "after-input" => "3",
                    _ => unreachable!("invariant: fixture suffixes are exhaustive"),
                };
                let (width, height) = if cine_rate_mode {
                    ([128, 64, 96][axis], 192)
                } else {
                    (256, 192)
                };
                json!({
                    "label": format!("{id}-{suffix}"),
                    "scope": "element",
                    "width": width,
                    "height": height,
                    "bytes": 100,
                    "sha256": digest_digit.repeat(64)
                })
            })
        }))
        .collect();
    let metrics = matches!(
        input_mode,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate)
    )
    .then(|| json!({"device_scale": {"device_pixel_ratio": 1.25}}));
    json!({
        "schema": 1,
        "status": "passed",
        "engine": "chromium",
        "bridge": "canvas",
        "revision": "0".repeat(40),
        "consumer_revision": "1".repeat(40),
        "actions": actions,
        "snapshots": snapshots,
        "screenshots": screenshots,
        "metrics": metrics,
        "cleanup": {
            "active_input_sources_released": true,
            "canvas_count": ids.len(),
            "canvas_attribute_names": input_mode.expected_attributes()
        }
    })
}

fn keyboard_action(id: &str, key: &str, code: &str, repeat: bool, release: bool) -> Value {
    let mut observed_events = vec![keyboard_event(id, key, code, "keydown", repeat)];
    if release {
        observed_events.push(keyboard_event(id, key, code, "keyup", false));
    }
    json!({
        "action": "trusted-keyboard",
        "canvas": id,
        "key": key,
        "code": code,
        "repeat": repeat,
        "focus": {"ok": true, "active_id": id},
        "observed_events": observed_events
    })
}

fn keyboard_event(id: &str, key: &str, code: &str, event_type: &str, repeat: bool) -> Value {
    json!({
        "type": event_type,
        "is_trusted": true,
        "target_id": id,
        "key": key,
        "code": code,
        "repeat": repeat,
        "alt_key": false,
        "ctrl_key": false,
        "meta_key": false,
        "shift_key": false
    })
}

fn fixture(ids: &[String]) -> TraceDocument {
    serde_json::from_value(fixture_value(ids)).expect("fixture is a valid trace document")
}

fn reject(value: Value, message: &str) {
    reject_with_mode(value, TraceInputMode::PointerWheel, message);
}

fn add_window_level_attributes(value: &mut Value) {
    for snapshot in value["snapshots"].as_array_mut().expect("snapshots array") {
        let attributes = snapshot["canvas"]["attributes"]
            .as_object_mut()
            .expect("attribute object");
        attributes.insert("data-ritk-window-center".to_owned(), json!("50"));
        attributes.insert("data-ritk-window-width".to_owned(), json!("100"));
        attributes.insert("data-ritk-window-preset-index".to_owned(), json!("0"));
    }
    let cleanup = value["cleanup"]["canvas_attribute_names"]
        .as_array_mut()
        .expect("cleanup attribute names");
    cleanup.extend([
        json!("data-ritk-window-center"),
        json!("data-ritk-window-width"),
        json!("data-ritk-window-preset-index"),
    ]);
}

fn add_interaction_attributes(value: &mut Value) {
    for snapshot in value["snapshots"].as_array_mut().expect("snapshots array") {
        let attributes = snapshot["canvas"]["attributes"]
            .as_object_mut()
            .expect("attribute object");
        attributes.insert("data-ritk-cine-enabled".to_owned(), json!("false"));
        attributes.insert("data-ritk-active-tool-index".to_owned(), json!("2"));
        attributes.insert("data-ritk-active-tool".to_owned(), json!("W/L"));
    }
    let cleanup = value["cleanup"]["canvas_attribute_names"]
        .as_array_mut()
        .expect("cleanup attribute names");
    cleanup.extend([
        json!("data-ritk-cine-enabled"),
        json!("data-ritk-active-tool-index"),
        json!("data-ritk-active-tool"),
    ]);
}

pub(super) fn reject_with_mode(value: Value, input_mode: TraceInputMode, message: &str) {
    let ids = default_ids();
    let document: TraceDocument = serde_json::from_value(value).expect("mutation keeps JSON shape");
    let error = validate_document(&document, &ids, input_mode)
        .expect_err("malformed fixture must be rejected");
    assert!(error.to_string().contains(message), "{error:#}");
}

#[test]
fn valid_three_canvas_trace_passes() {
    let ids = default_ids();
    let report =
        validate_document(&fixture(&ids), &ids, TraceInputMode::PointerWheel).expect("valid trace");
    assert_eq!(report.engine, "chromium");
    assert_eq!(report.canvas_count, 3);
}

#[test]
fn committed_manual_fixture_passes_the_same_validator() {
    let document: TraceDocument = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/browser-trace.json"
    )))
    .expect("committed browser trace fixture");
    let ids = default_ids();
    validate_document(&document, &ids, TraceInputMode::PointerWheel)
        .expect("committed browser trace fixture is valid");
}

#[test]
fn custom_canvas_ids_are_checked_in_order() {
    let ids = ["axial-view", "coronal-view", "sagittal-view"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let report = validate_document(&fixture(&ids), &ids, TraceInputMode::PointerWheel)
        .expect("custom ids are valid");
    assert_eq!(report.canvas_count, 3);
}

#[test]
fn status_and_revisions_are_required() {
    let mut value = fixture_value(&default_ids());
    value["status"] = json!("failed");
    reject(value, "status");

    let mut value = fixture_value(&default_ids());
    value["consumer_revision"] = json!("short");
    reject(value, "RITK revision");
}

#[test]
fn semantic_attributes_and_dimensions_are_checked() {
    let mut value = fixture_value(&default_ids());
    value["snapshots"][2]["canvas"]["attributes"]["data-ritk-axis"] = json!("0");
    reject(value, "axis");

    let mut value = fixture_value(&default_ids());
    value["snapshots"][1]["canvas"]["width"] = json!(0);
    reject(value, "intrinsic width");

    let mut value = fixture_value(&default_ids());
    value["snapshots"][0]["canvas"]["attributes"]
        .as_object_mut()
        .expect("attribute object")
        .remove("data-ritk-frame-height");
    reject(value, "complete RITK attribute set");
}

#[test]
fn complete_window_level_attribute_extension_passes() {
    let mut value = fixture_value(&default_ids());
    add_window_level_attributes(&mut value);
    let document: TraceDocument = serde_json::from_value(value).expect("window fixture shape");
    validate_document(&document, &default_ids(), TraceInputMode::PointerWheel)
        .expect("complete window-level attribute extension is valid");
}

#[test]
fn complete_interaction_attribute_extension_passes() {
    let mut value = fixture_value(&default_ids());
    add_interaction_attributes(&mut value);
    let document: TraceDocument = serde_json::from_value(value).expect("interaction fixture shape");
    validate_document(&document, &default_ids(), TraceInputMode::PointerWheel)
        .expect("complete interaction attribute extension is valid");
}
