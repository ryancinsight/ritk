use super::*;
use serde_json::{json, Value};

fn default_ids() -> Vec<String> {
    DEFAULT_CANVAS_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect()
}

fn fixture_value(ids: &[String]) -> Value {
    fixture_value_for_mode(ids, TraceInputMode::PointerWheel)
}

fn keyboard_fixture_value(ids: &[String]) -> Value {
    fixture_value_for_mode(ids, TraceInputMode::PointerWheelKeyboard)
}

fn fixture_value_for_mode(ids: &[String], input_mode: TraceInputMode) -> Value {
    let attributes = |axis: usize, slice_index: usize| {
        json!({
            "data-ritk-load-state": "ready",
            "data-ritk-frame-state": "presented",
            "data-ritk-axis": axis.to_string(),
            "data-ritk-slice-index": slice_index.to_string(),
            "data-ritk-slice-count": "4",
            "data-ritk-frame-width": "256",
            "data-ritk-frame-height": "192"
        })
    };
    let mut actions = Vec::new();
    for id in ids {
        if matches!(input_mode, TraceInputMode::PointerWheelKeyboard) {
            actions.push(json!({
                "action": "trusted-keyboard",
                "canvas": id,
                "key": KEYBOARD_TRACE_KEY,
                "code": KEYBOARD_TRACE_KEY,
                "repeat": false,
                "focus": {"ok": true, "active_id": id},
                "observed_events": [
                    {
                        "type": "keydown",
                        "is_trusted": true,
                        "target_id": id,
                        "key": KEYBOARD_TRACE_KEY,
                        "code": KEYBOARD_TRACE_KEY,
                        "repeat": false,
                        "alt_key": false,
                        "ctrl_key": false,
                        "meta_key": false,
                        "shift_key": false
                    },
                    {
                        "type": "keyup",
                        "is_trusted": true,
                        "target_id": id,
                        "key": KEYBOARD_TRACE_KEY,
                        "code": KEYBOARD_TRACE_KEY,
                        "repeat": false,
                        "alt_key": false,
                        "ctrl_key": false,
                        "meta_key": false,
                        "shift_key": false
                    }
                ]
            }));
        }
        actions.push(json!({"action": "trusted-pointer-drag", "canvas": id}));
        actions.push(json!({"action": "trusted-wheel", "canvas": id}));
    }
    let snapshots: Vec<Value> = ids
        .iter()
        .enumerate()
        .flat_map(|(axis, id)| {
            let initial = json!({
                "label": format!("{id}-initial"),
                "canvas": {"id": id, "width": 256, "height": 192, "attributes": attributes(axis, 0)}
            });
            let after_input_index = if matches!(input_mode, TraceInputMode::PointerWheelKeyboard) {
                2
            } else {
                1
            };
            let after_input = json!({
                "label": format!("{id}-after-input"),
                "canvas": {"id": id, "width": 256, "height": 192, "attributes": attributes(axis, after_input_index)}
            });
            if matches!(input_mode, TraceInputMode::PointerWheelKeyboard) {
                vec![
                    initial,
                    json!({
                        "label": format!("{id}-after-keyboard"),
                        "canvas": {"id": id, "width": 256, "height": 192, "attributes": attributes(axis, 1)}
                    }),
                    after_input,
                ]
            } else {
                vec![initial, after_input]
            }
        })
        .collect();
    let screenshots: Vec<Value> = [
            json!({"label": "window-initial", "width": 1280, "height": 720, "bytes": 100, "sha256": "0".repeat(64)}),
            json!({"label": "window-final", "width": 1280, "height": 720, "bytes": 100, "sha256": "1".repeat(64)}),
        ]
        .into_iter()
        .chain(ids.iter().flat_map(|id| {
            [
                json!({"label": format!("{id}-initial"), "scope": "element", "width": 256, "height": 192, "bytes": 100, "sha256": "2".repeat(64)}),
                json!({"label": format!("{id}-after-input"), "scope": "element", "width": 256, "height": 192, "bytes": 100, "sha256": "3".repeat(64)}),
            ]
        }))
        .collect();
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
        "cleanup": {
            "active_input_sources_released": true,
            "canvas_count": ids.len(),
            "canvas_attribute_names": EXPECTED_ATTRIBUTES
        }
    })
}

fn fixture(ids: &[String]) -> TraceDocument {
    serde_json::from_value(fixture_value(ids)).expect("fixture is a valid trace document")
}

fn reject(value: Value, message: &str) {
    reject_with_mode(value, TraceInputMode::PointerWheel, message);
}

fn reject_with_mode(value: Value, input_mode: TraceInputMode, message: &str) {
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
fn trusted_wheel_must_advance_multi_slice_canvas() {
    let mut value = fixture_value(&default_ids());
    for snapshot in value["snapshots"].as_array_mut().expect("snapshots array") {
        if snapshot["label"]
            .as_str()
            .expect("snapshot label")
            .ends_with("after-input")
        {
            snapshot["canvas"]["attributes"]["data-ritk-slice-index"] = json!("0");
        }
    }
    reject(value, "did not advance its multi-slice index");
}

#[test]
fn singleton_slice_canvas_may_remain_at_the_same_index() {
    let mut value = fixture_value(&default_ids());
    for snapshot in value["snapshots"].as_array_mut().expect("snapshots array") {
        snapshot["canvas"]["attributes"]["data-ritk-slice-count"] = json!("1");
        snapshot["canvas"]["attributes"]["data-ritk-slice-index"] = json!("0");
    }
    let ids = default_ids();
    let document: TraceDocument = serde_json::from_value(value).expect("singleton fixture");
    validate_document(&document, &ids, TraceInputMode::PointerWheel)
        .expect("singleton canvas is valid");
}

#[test]
fn actions_screenshots_and_cleanup_are_checked() {
    let mut value = fixture_value(&default_ids());
    value["actions"]
        .as_array_mut()
        .expect("actions array")
        .pop();
    reject(value, "actions");

    let mut value = fixture_value(&default_ids());
    value["screenshots"][2]["scope"] = json!("window");
    reject(value, "invalid scope");

    let mut value = fixture_value(&default_ids());
    value["cleanup"]["active_input_sources_released"] = json!(false);
    reject(value, "release");
}

#[test]
fn file_validation_reports_the_consumer_and_engine() {
    let ids = default_ids();
    let file = tempfile::NamedTempFile::new().expect("temporary trace file");
    fs::write(
        file.path(),
        serde_json::to_vec(&fixture_value(&ids)).expect("serialize fixture"),
    )
    .expect("write fixture");
    let report = validate_file(file.path(), &[], TraceInputMode::PointerWheel)
        .expect("file fixture is valid");
    assert_eq!(report.consumer_revision, "1".repeat(40));
}

#[test]
fn focused_keyboard_evidence_passes_and_untrusted_events_fail() {
    let ids = default_ids();
    let value = keyboard_fixture_value(&ids);
    let document: TraceDocument = serde_json::from_value(value.clone()).expect("keyboard fixture");
    validate_document(&document, &ids, TraceInputMode::PointerWheelKeyboard)
        .expect("focused keyboard evidence is valid");
    assert_eq!(
        value["snapshots"]
            .as_array()
            .expect("snapshot array")
            .iter()
            .map(|snapshot| snapshot["label"].as_str().expect("snapshot label"))
            .collect::<Vec<_>>(),
        vec![
            "ritk-snap-axial-initial",
            "ritk-snap-axial-after-keyboard",
            "ritk-snap-axial-after-input",
            "ritk-snap-coronal-initial",
            "ritk-snap-coronal-after-keyboard",
            "ritk-snap-coronal-after-input",
            "ritk-snap-sagittal-initial",
            "ritk-snap-sagittal-after-keyboard",
            "ritk-snap-sagittal-after-input",
        ]
    );

    let mut untrusted = value;
    untrusted["actions"][0]["observed_events"][0]["is_trusted"] = json!(false);
    reject_with_mode(
        untrusted,
        TraceInputMode::PointerWheelKeyboard,
        "was not trusted",
    );

    let mut missing_focus = keyboard_fixture_value(&ids);
    missing_focus["actions"][0]["focus"] = Value::Null;
    reject_with_mode(
        missing_focus,
        TraceInputMode::PointerWheelKeyboard,
        "missing focus evidence",
    );
}

#[test]
fn keyboard_wheel_progression_uses_post_keyboard_snapshot() {
    let mut value = keyboard_fixture_value(&default_ids());
    for snapshot in value["snapshots"].as_array_mut().expect("snapshots array") {
        if snapshot["label"]
            .as_str()
            .expect("snapshot label")
            .ends_with("after-input")
        {
            snapshot["canvas"]["attributes"]["data-ritk-slice-index"] = json!("1");
        }
    }
    reject_with_mode(
        value,
        TraceInputMode::PointerWheelKeyboard,
        "did not advance its multi-slice index",
    );
}

#[test]
fn keyboard_mode_requires_the_intermediate_snapshot() {
    let mut value = keyboard_fixture_value(&default_ids());
    value["snapshots"][1]["label"] = json!("ritk-snap-axial-after-input");
    reject_with_mode(
        value,
        TraceInputMode::PointerWheelKeyboard,
        "repeats snapshot label",
    );
}

#[test]
fn keyboard_mode_requires_one_keyboard_action_per_canvas() {
    let ids = default_ids();
    let document: TraceDocument =
        serde_json::from_value(fixture_value(&ids)).expect("pointer fixture");
    let error = validate_document(&document, &ids, TraceInputMode::PointerWheelKeyboard)
        .expect_err("keyboard mode must reject a pointer-only trace");
    assert!(error.to_string().contains("expected 9"), "{error:#}");
}

#[test]
fn oversized_trace_is_rejected_before_json_allocation() {
    let file = tempfile::NamedTempFile::new().expect("temporary trace file");
    let length = usize::try_from(MAX_TRACE_BYTES).expect("trace bound fits this platform") + 1;
    fs::write(file.path(), vec![b' '; length]).expect("write oversized trace");
    let error = validate_file(file.path(), &[], TraceInputMode::PointerWheel)
        .expect_err("oversized trace must fail");
    assert!(error.to_string().contains("limit"), "{error:#}");
}
