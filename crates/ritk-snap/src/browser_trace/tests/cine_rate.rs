use serde_json::{json, Value};

use super::super::{validate_document, KeyboardTraceKind, TraceDocument, TraceInputMode};
use super::{cine_rate_fixture_value, default_ids, reject_with_mode};

fn mode() -> TraceInputMode {
    TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate)
}

fn reject_cine(value: Value, message: &str) {
    reject_with_mode(value, mode(), message);
}

#[test]
fn complete_cine_sequence_passes() {
    let ids = default_ids();
    let value = cine_rate_fixture_value(&ids);
    let document: TraceDocument =
        serde_json::from_value(value.clone()).expect("cine fixture shape");
    validate_document(&document, &ids, mode()).expect("complete cine sequence is valid");
    assert_eq!(
        value["snapshots"][0]["canvas"]["attributes"]["data-ritk-display-aspect"],
        (2.0_f64 / 3.0).to_string()
    );
    assert_eq!(
        value["snapshots"][0]["canvas"]["css_width"],
        json!((2.0_f64 / 3.0) * 153.2)
    );
}

#[test]
fn fractional_css_dimensions_accept_nonunit_device_scale_rounding() {
    let ids = default_ids();
    let mut value = cine_rate_fixture_value(&ids);
    for screenshot in value["screenshots"]
        .as_array_mut()
        .expect("screenshots array")
    {
        if screenshot["scope"].as_str() == Some("element") {
            let label = screenshot["label"].as_str().expect("screenshot label");
            screenshot["width"] = json!(if label.starts_with("ritk-snap-axial-") {
                127
            } else if label.starts_with("ritk-snap-coronal-") {
                63
            } else {
                95
            });
            screenshot["height"] = json!(191);
        }
    }
    let document: TraceDocument = serde_json::from_value(value).expect("cine fixture shape");
    validate_document(&document, &ids, mode())
        .expect("floor-rounded element captures at device scale 1.25 are valid");
}

#[test]
fn cine_css_box_accepts_independent_dimension_quantization() {
    let mut value = cine_rate_fixture_value(&default_ids());
    // Exercise both one-pixel terms just inside the analytical boundary without
    // making binary floating-point equality part of the test contract.
    let quantization = 0.99;
    for snapshot in value["snapshots"]
        .as_array_mut()
        .expect("snapshots array")
        .iter_mut()
        .filter(|snapshot| snapshot["canvas"]["id"].as_str() == Some("ritk-snap-axial"))
    {
        let canvas = &mut snapshot["canvas"];
        canvas["css_width"] = json!((2.0_f64 / 3.0) * 153.2 + quantization);
        canvas["css_height"] = json!(153.2 - quantization);
    }
    for screenshot in value["screenshots"]
        .as_array_mut()
        .expect("screenshots array")
        .iter_mut()
        .filter(|screenshot| {
            screenshot["scope"].as_str() == Some("element")
                && screenshot["label"]
                    .as_str()
                    .is_some_and(|label| label.starts_with("ritk-snap-axial-"))
        })
    {
        screenshot["width"] = json!(129);
        screenshot["height"] = json!(190);
    }
    let document: TraceDocument = serde_json::from_value(value).expect("cine fixture shape");
    validate_document(&document, &default_ids(), mode())
        .expect("independently quantized CSS dimensions are valid");
}

#[test]
fn cine_snapshots_require_positive_finite_display_aspect() {
    let mut missing = cine_rate_fixture_value(&default_ids());
    missing["snapshots"][0]["canvas"]["attributes"]
        .as_object_mut()
        .expect("attribute object")
        .remove("data-ritk-display-aspect");
    reject_cine(
        missing,
        "missing non-null attribute \"data-ritk-display-aspect\"",
    );

    for invalid in ["invalid", "-1", "0", "NaN", "inf"] {
        let mut value = cine_rate_fixture_value(&default_ids());
        value["snapshots"][0]["canvas"]["attributes"]["data-ritk-display-aspect"] = json!(invalid);
        reject_cine(value, "display aspect");
    }
}

#[test]
fn cine_css_box_must_preserve_physical_not_pixel_aspect() {
    let mut squashed = cine_rate_fixture_value(&default_ids());
    for snapshot in squashed["snapshots"]
        .as_array_mut()
        .expect("snapshots array")
        .iter_mut()
        .filter(|snapshot| snapshot["canvas"]["id"].as_str() == Some("ritk-snap-axial"))
    {
        snapshot["canvas"]["css_width"] = json!(153.2);
    }
    for screenshot in squashed["screenshots"]
        .as_array_mut()
        .expect("screenshots array")
        .iter_mut()
        .filter(|screenshot| {
            screenshot["scope"].as_str() == Some("element")
                && screenshot["label"]
                    .as_str()
                    .is_some_and(|label| label.starts_with("ritk-snap-axial-"))
        })
    {
        // Keep the pre-existing screenshot/CSS/device-scale guard satisfied so
        // only the independent physical-aspect oracle detects the distortion.
        screenshot["width"] = json!(192);
        screenshot["height"] = json!(192);
    }
    reject_cine(squashed, "do not preserve its physical display aspect");
}

#[test]
fn actions_require_exact_per_canvas_sequence_and_target() {
    let mut reordered = cine_rate_fixture_value(&default_ids());
    reordered["actions"]
        .as_array_mut()
        .expect("actions array")
        .swap(4, 5);
    reject_cine(reordered, "action sequence expected");

    let mut wrong_target = cine_rate_fixture_value(&default_ids());
    wrong_target["actions"][2]["canvas"] = json!("ritk-snap-coronal");
    reject_cine(wrong_target, "targeting");
}

#[test]
fn action_metadata_and_focus_are_exact() {
    let mut wrong_code = cine_rate_fixture_value(&default_ids());
    wrong_code["actions"][0]["code"] = json!("Minus");
    reject_cine(wrong_code, "invalid key/code/repeat metadata");

    let mut wrong_repeat = cine_rate_fixture_value(&default_ids());
    wrong_repeat["actions"][1]["repeat"] = json!(false);
    reject_cine(wrong_repeat, "invalid key/code/repeat metadata");

    let mut wrong_focus = cine_rate_fixture_value(&default_ids());
    wrong_focus["actions"][2]["focus"]["active_id"] = json!("ritk-snap-coronal");
    reject_cine(wrong_focus, "did not focus its target");
}

#[test]
fn nonrepeat_and_repeat_event_shapes_are_exact() {
    let mut nonrepeat_release = cine_rate_fixture_value(&default_ids());
    let release = nonrepeat_release["actions"][1]["observed_events"][1].clone();
    nonrepeat_release["actions"][0]["observed_events"]
        .as_array_mut()
        .expect("observed event array")
        .push(release);
    reject_cine(nonrepeat_release, "must observe one keydown");

    let mut missing_repeat_release = cine_rate_fixture_value(&default_ids());
    missing_repeat_release["actions"][1]["observed_events"]
        .as_array_mut()
        .expect("observed event array")
        .pop();
    reject_cine(missing_repeat_release, "followed by non-repeated keyup");

    let mut repeated_keyup = cine_rate_fixture_value(&default_ids());
    repeated_keyup["actions"][1]["observed_events"][1]["repeat"] = json!(true);
    reject_cine(
        repeated_keyup,
        "invalid trusted target/key/code/repeat metadata",
    );
}

#[test]
fn every_observed_event_is_trusted_unmodified_and_targeted() {
    let mut untrusted = cine_rate_fixture_value(&default_ids());
    untrusted["actions"][1]["observed_events"][0]["is_trusted"] = json!(false);
    reject_cine(untrusted, "invalid trusted target/key/code/repeat metadata");

    let mut wrong_target = cine_rate_fixture_value(&default_ids());
    wrong_target["actions"][3]["observed_events"][1]["target_id"] = json!("ritk-snap-coronal");
    reject_cine(
        wrong_target,
        "invalid trusted target/key/code/repeat metadata",
    );

    let mut modified = cine_rate_fixture_value(&default_ids());
    modified["actions"][2]["observed_events"][0]["shift_key"] = json!(true);
    reject_cine(modified, "active shift modifier");
}

#[test]
fn snapshots_require_all_six_stages_in_order() {
    let mut reordered = cine_rate_fixture_value(&default_ids());
    reordered["snapshots"]
        .as_array_mut()
        .expect("snapshots array")
        .swap(1, 2);
    reject_cine(reordered, "snapshot sequence expected");

    let mut missing = cine_rate_fixture_value(&default_ids());
    missing["snapshots"]
        .as_array_mut()
        .expect("snapshots array")
        .pop();
    reject_cine(missing, "expected 18");
}

#[test]
fn rates_are_integer_bounded_and_follow_exact_deltas() {
    let mut fractional = cine_rate_fixture_value(&default_ids());
    fractional["snapshots"][0]["canvas"]["attributes"]["data-ritk-cine-fps"] = json!("12.5");
    reject_cine(fractional, "invalid cine FPS");

    for rate in ["1", "60"] {
        let mut boundary = cine_rate_fixture_value(&default_ids());
        boundary["snapshots"][0]["canvas"]["attributes"]["data-ritk-cine-fps"] = json!(rate);
        reject_cine(boundary, "integer from 2 through 59");
    }

    for (stage, rate) in [(1, "14"), (2, "12"), (3, "11"), (4, "13"), (5, "13")] {
        let mut wrong_delta = cine_rate_fixture_value(&default_ids());
        wrong_delta["snapshots"][stage]["canvas"]["attributes"]["data-ritk-cine-fps"] = json!(rate);
        reject_cine(wrong_delta, "change by exactly +1 and -1");
    }
}

#[test]
fn generations_advance_only_for_effective_rate_actions() {
    let mut zero = cine_rate_fixture_value(&default_ids());
    zero["snapshots"][0]["canvas"]["attributes"]["data-ritk-frame-generation"] = json!("0");
    reject_cine(zero, "zero frame generation");

    for (stage, generation) in [(1, "9"), (2, "9"), (3, "8"), (4, "10")] {
        let mut wrong = cine_rate_fixture_value(&default_ids());
        wrong["snapshots"][stage]["canvas"]["attributes"]["data-ritk-frame-generation"] =
            json!(generation);
        reject_cine(wrong, "advance once for effective rate actions");
    }

    let mut stale_input = cine_rate_fixture_value(&default_ids());
    stale_input["snapshots"][5]["canvas"]["attributes"]["data-ritk-frame-generation"] = json!("9");
    reject_cine(stale_input, "fresh frame after pointer and wheel input");
}

#[test]
fn rate_stages_preserve_slice_dimensions_and_presented_state() {
    let mut changed_slice = cine_rate_fixture_value(&default_ids());
    changed_slice["snapshots"][3]["canvas"]["attributes"]["data-ritk-slice-index"] = json!("1");
    reject_cine(changed_slice, "changed its slice during a cine-rate action");

    let mut changed_count = cine_rate_fixture_value(&default_ids());
    changed_count["snapshots"][2]["canvas"]["attributes"]["data-ritk-slice-count"] = json!("5");
    reject_cine(changed_count, "changed its slice during a cine-rate action");

    let mut changed_dimensions = cine_rate_fixture_value(&default_ids());
    changed_dimensions["snapshots"][4]["canvas"]["width"] = json!(255);
    changed_dimensions["snapshots"][4]["canvas"]["attributes"]["data-ritk-frame-width"] =
        json!("255");
    reject_cine(
        changed_dimensions,
        "changed intrinsic dimensions during a cine-rate action",
    );

    let mut empty_stage = cine_rate_fixture_value(&default_ids());
    empty_stage["snapshots"][2]["canvas"]["attributes"]["data-ritk-load-state"] = json!("empty");
    reject_cine(empty_stage, "inconsistent presented frame dimensions");
}

#[test]
fn wheel_progression_uses_the_post_decrease_repeat_snapshot() {
    let mut unchanged = cine_rate_fixture_value(&default_ids());
    unchanged["snapshots"][5]["canvas"]["attributes"]["data-ritk-slice-index"] = json!("0");
    reject_cine(unchanged, "did not advance its multi-slice index");
}

#[test]
fn screenshots_cover_each_stage_with_stable_dimensions_and_repeat_hashes() {
    let mut missing = cine_rate_fixture_value(&default_ids());
    missing["screenshots"]
        .as_array_mut()
        .expect("screenshots array")
        .pop();
    reject_cine(missing, "expected 20");

    let mut invalid_digest = cine_rate_fixture_value(&default_ids());
    invalid_digest["screenshots"][3]["sha256"] = json!("not-a-digest");
    reject_cine(invalid_digest, "invalid SHA-256 digest");

    let mut resized = cine_rate_fixture_value(&default_ids());
    resized["screenshots"][4]["width"] = json!(254);
    reject_cine(resized, "dimensions do not match its CSS box");

    let mut constant_wrong_dimensions = cine_rate_fixture_value(&default_ids());
    for screenshot in constant_wrong_dimensions["screenshots"]
        .as_array_mut()
        .expect("screenshots array")
    {
        if screenshot["scope"].as_str() == Some("element") {
            screenshot["width"] = json!(1);
            screenshot["height"] = json!(1);
        }
    }
    reject_cine(
        constant_wrong_dimensions,
        "dimensions do not match its CSS box",
    );

    let mut changed_repeat = cine_rate_fixture_value(&default_ids());
    changed_repeat["screenshots"][4]["sha256"] = json!("6".repeat(64));
    reject_cine(
        changed_repeat,
        "repeated cine input changed its screenshot hash",
    );
}

#[test]
fn cine_screenshots_require_independent_device_scale_metrics() {
    let mut missing = cine_rate_fixture_value(&default_ids());
    missing["metrics"] = Value::Null;
    reject_cine(missing, "missing device-scale metrics");

    let mut missing_scale = cine_rate_fixture_value(&default_ids());
    missing_scale["metrics"]["device_scale"] = Value::Null;
    reject_cine(missing_scale, "missing device-scale metrics");

    for ratio in [json!(0.49), json!(4.01)] {
        let mut invalid = cine_rate_fixture_value(&default_ids());
        invalid["metrics"]["device_scale"]["device_pixel_ratio"] = ratio;
        reject_cine(invalid, "device pixel ratio");
    }
}
