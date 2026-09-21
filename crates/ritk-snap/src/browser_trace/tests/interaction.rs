//! Browser interaction and evidence validation tests.

use super::*;

#[test]
fn complete_window_and_interaction_extensions_pass_together() {
    let mut value = fixture_value(&default_ids());
    add_window_level_attributes(&mut value);
    add_interaction_attributes(&mut value);
    let document: TraceDocument = serde_json::from_value(value).expect("combined fixture shape");
    validate_document(&document, &default_ids(), TraceInputMode::PointerWheel)
        .expect("combined attribute extensions are valid");
}

#[test]
fn cine_rate_accepts_window_and_interaction_extensions() {
    let ids = default_ids();
    let mut value = cine_rate_fixture_value(&ids);
    add_window_level_attributes(&mut value);
    add_interaction_attributes(&mut value);
    let document: TraceDocument =
        serde_json::from_value(value).expect("combined cine fixture shape");
    validate_document(
        &document,
        &ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate),
    )
    .expect("combined cine attributes are valid");
}

#[test]
fn window_level_attribute_extension_rejects_partial_or_unknown_sets() {
    let mut partial = fixture_value(&default_ids());
    add_window_level_attributes(&mut partial);
    partial["snapshots"][0]["canvas"]["attributes"]
        .as_object_mut()
        .expect("attribute object")
        .remove("data-ritk-window-width");
    reject(partial, "complete RITK attribute set");

    let mut unknown = fixture_value(&default_ids());
    add_window_level_attributes(&mut unknown);
    unknown["snapshots"][0]["canvas"]["attributes"]["data-ritk-window-extra"] = json!("1");
    reject(unknown, "complete RITK attribute set");

    let mut partial_interaction = fixture_value(&default_ids());
    add_interaction_attributes(&mut partial_interaction);
    partial_interaction["snapshots"][0]["canvas"]["attributes"]
        .as_object_mut()
        .expect("attribute object")
        .remove("data-ritk-active-tool");
    reject(partial_interaction, "complete RITK attribute set");
}

#[test]
fn window_level_values_are_finite_positive_and_indexed() {
    for (name, value, message) in [
        ("data-ritk-window-center", "NaN", "non-finite window center"),
        (
            "data-ritk-window-width",
            "0",
            "non-positive or non-finite window width",
        ),
        (
            "data-ritk-window-preset-index",
            "1.5",
            "invalid window preset index",
        ),
    ] {
        let mut invalid = fixture_value(&default_ids());
        add_window_level_attributes(&mut invalid);
        invalid["snapshots"][0]["canvas"]["attributes"][name] = json!(value);
        reject(invalid, message);
    }
}

#[test]
fn interaction_attribute_values_are_validated() {
    for (name, value, message) in [
        (
            "data-ritk-cine-enabled",
            "enabled",
            "invalid cine-enabled value",
        ),
        (
            "data-ritk-active-tool-index",
            "tool",
            "invalid active tool index value",
        ),
    ] {
        let mut invalid = fixture_value(&default_ids());
        add_interaction_attributes(&mut invalid);
        invalid["snapshots"][0]["canvas"]["attributes"][name] = json!(value);
        reject(invalid, message);
    }

    let mut invalid = fixture_value(&default_ids());
    add_interaction_attributes(&mut invalid);
    invalid["snapshots"][0]["canvas"]["attributes"]["data-ritk-active-tool"] = json!("");
    reject(invalid, "empty attribute");
}

#[test]
fn linked_cursor_attributes_are_bounded_and_orientation_typed() {
    let mut invalid = fixture_value(&default_ids());
    invalid["snapshots"][0]["canvas"]["attributes"]["data-ritk-crosshair-visible"] = json!("yes");
    reject(invalid, "invalid crosshair visibility");

    let mut invalid = fixture_value(&default_ids());
    invalid["snapshots"][0]["canvas"]["attributes"]["data-ritk-linked-cursor"] = json!("1,2");
    reject(invalid, "invalid linked cursor");

    let mut invalid = fixture_value(&default_ids());
    invalid["snapshots"][0]["canvas"]["attributes"]["data-ritk-view-rotation"] = json!("45");
    reject(invalid, "invalid view rotation");
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
    validate_document(
        &document,
        &ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
    )
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
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
        "was not trusted",
    );

    let mut missing_focus = keyboard_fixture_value(&ids);
    missing_focus["actions"][0]["focus"] = Value::Null;
    reject_with_mode(
        missing_focus,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
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
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
        "did not advance its multi-slice index",
    );
}

#[test]
fn keyboard_mode_requires_the_intermediate_snapshot() {
    let mut value = keyboard_fixture_value(&default_ids());
    value["snapshots"][1]["label"] = json!("ritk-snap-axial-after-input");
    reject_with_mode(
        value,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
        "repeats snapshot label",
    );
}

#[test]
fn keyboard_mode_requires_one_keyboard_action_per_canvas() {
    let ids = default_ids();
    let document: TraceDocument =
        serde_json::from_value(fixture_value(&ids)).expect("pointer fixture");
    let error = validate_document(
        &document,
        &ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation),
    )
    .expect_err("keyboard mode must reject a pointer-only trace");
    assert!(error.to_string().contains("expected 9"), "{error:#}");
}

#[test]
fn cine_rate_keyboard_requires_equal_and_an_increased_rate() {
    let ids = default_ids();
    let value = cine_rate_fixture_value(&ids);
    let document: TraceDocument = serde_json::from_value(value.clone()).expect("cine rate fixture");
    validate_document(
        &document,
        &ids,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate),
    )
    .expect("cine rate evidence is valid");
    assert_eq!(
        value["snapshots"][0]["canvas"]["attributes"]["data-ritk-cine-fps"],
        "12"
    );
    assert_eq!(
        value["snapshots"][1]["canvas"]["attributes"]["data-ritk-cine-fps"],
        "13"
    );

    let mut unchanged = value.clone();
    unchanged["snapshots"][1]["canvas"]["attributes"]["data-ritk-cine-fps"] = json!("12");
    reject_with_mode(
        unchanged,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate),
        "change by exactly +1 and -1",
    );

    let mut wrong_key = value;
    wrong_key["actions"][0]["key"] = json!("ArrowDown");
    reject_with_mode(
        wrong_key,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate),
        "invalid key/code",
    );
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
