//! Validation for browser-trace screenshots and teardown evidence.

use anyhow::{bail, Result};
use std::collections::BTreeSet;

use super::{cine_rate, TraceCleanup, TraceInputMode, TraceScreenshot, TraceSnapshot};

/// Validate the window and canvas screenshot manifest.
pub(super) fn validate_screenshots(
    screenshots: &[TraceScreenshot],
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
    device_pixel_ratio: Option<f64>,
) -> Result<()> {
    let suffixes = match input_mode {
        TraceInputMode::PointerWheel
        | TraceInputMode::PointerWheelKeyboard(super::KeyboardTraceKind::Navigation) => {
            ["initial", "after-input"].as_slice()
        }
        TraceInputMode::PointerWheelKeyboard(super::KeyboardTraceKind::CineRate) => {
            cine_rate::SNAPSHOT_SUFFIXES.as_slice()
        }
    };
    let expected_screenshot_count = 2 + canvas_ids.len() * suffixes.len();
    if screenshots.len() != expected_screenshot_count {
        bail!(
            "browser trace contains {} screenshots; expected {expected_screenshot_count}",
            screenshots.len()
        )
    }

    let mut labels = BTreeSet::new();
    for screenshot in screenshots {
        let expected_scope =
            if matches!(screenshot.label.as_str(), "window-initial" | "window-final") {
                None
            } else if canvas_ids.iter().any(|id| {
                suffixes
                    .iter()
                    .any(|suffix| screenshot.label == format!("{id}-{suffix}"))
            }) {
                Some("element")
            } else {
                bail!(
                    "browser trace contains unknown screenshot label {:?}",
                    screenshot.label
                )
            };
        if !labels.insert(screenshot.label.clone()) {
            bail!(
                "browser trace repeats screenshot label {:?}",
                screenshot.label
            )
        }
        if screenshot.scope.as_deref() != expected_scope {
            bail!("screenshot {:?} has an invalid scope", screenshot.label)
        }
        if screenshot.width == 0 || screenshot.height == 0 || screenshot.bytes == 0 {
            bail!(
                "screenshot {:?} has invalid dimensions or byte count",
                screenshot.label
            )
        }
        if screenshot.sha256.len() != 64
            || !screenshot
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            bail!(
                "screenshot {:?} has an invalid SHA-256 digest",
                screenshot.label
            )
        }
        if expected_scope == Some("element") {
            let Some(snapshot) = snapshots
                .iter()
                .find(|snapshot| snapshot.label == screenshot.label)
            else {
                bail!(
                    "element screenshot {:?} has no matching canvas snapshot",
                    screenshot.label
                )
            };
            if let Some(ratio) = device_pixel_ratio {
                validate_css_capture_dimensions(screenshot, snapshot, ratio)?;
            }
        }
    }

    let mut expected_labels =
        BTreeSet::from(["window-initial".to_owned(), "window-final".to_owned()]);
    for id in canvas_ids {
        for suffix in suffixes {
            expected_labels.insert(format!("{id}-{suffix}"));
        }
    }
    if labels != expected_labels {
        bail!(
            "browser trace screenshot labels do not cover the required window and canvas captures"
        )
    }
    for id in canvas_ids {
        let initial_label = format!("{id}-initial");
        let initial = screenshots
            .iter()
            .find(|screenshot| screenshot.label == initial_label)
            .expect("invariant: required screenshot labels were established above");
        for suffix in suffixes.iter().skip(1) {
            let label = format!("{id}-{suffix}");
            let stage = screenshots
                .iter()
                .find(|screenshot| screenshot.label == label)
                .expect("invariant: required screenshot labels were established above");
            if (stage.width, stage.height) != (initial.width, initial.height) {
                bail!("canvas {id:?} element screenshot dimensions changed between stages")
            }
        }
        if matches!(
            input_mode,
            TraceInputMode::PointerWheelKeyboard(super::KeyboardTraceKind::CineRate)
        ) {
            validate_repeat_hash(screenshots, id, "after-keyboard", "after-repeat")?;
            validate_repeat_hash(screenshots, id, "after-decrease", "after-decrease-repeat")?;
        }
    }
    Ok(())
}

fn validate_css_capture_dimensions(
    screenshot: &TraceScreenshot,
    snapshot: &TraceSnapshot,
    device_pixel_ratio: f64,
) -> Result<()> {
    let expected_width = snapshot.canvas.css_width * device_pixel_ratio;
    let expected_height = snapshot.canvas.css_height * device_pixel_ratio;
    if !matches_rounded_dimension(screenshot.width, expected_width)
        || !matches_rounded_dimension(screenshot.height, expected_height)
    {
        bail!(
            "element screenshot {:?} dimensions do not match its CSS box at the measured device pixel ratio",
            screenshot.label
        )
    }
    Ok(())
}

fn matches_rounded_dimension(actual: u32, expected: f64) -> bool {
    let actual = f64::from(actual);
    actual == expected.floor() || actual == expected.ceil()
}

fn validate_repeat_hash(
    screenshots: &[TraceScreenshot],
    canvas_id: &str,
    preceding_suffix: &str,
    repeat_suffix: &str,
) -> Result<()> {
    let preceding_label = format!("{canvas_id}-{preceding_suffix}");
    let repeat_label = format!("{canvas_id}-{repeat_suffix}");
    let preceding = screenshots
        .iter()
        .find(|screenshot| screenshot.label == preceding_label)
        .expect("invariant: required screenshot labels were established above");
    let repeat = screenshots
        .iter()
        .find(|screenshot| screenshot.label == repeat_label)
        .expect("invariant: required screenshot labels were established above");
    if repeat.sha256 != preceding.sha256 {
        bail!("canvas {canvas_id:?} repeated cine input changed its screenshot hash")
    }
    Ok(())
}

/// Validate the input-release and canvas-attribute teardown evidence.
pub(super) fn validate_cleanup(
    cleanup: &TraceCleanup,
    canvas_ids: &[String],
    expected_attributes: &[&str],
) -> Result<()> {
    if !cleanup.active_input_sources_released {
        bail!("browser trace did not release active input sources")
    }
    if cleanup.canvas_count != canvas_ids.len() {
        bail!(
            "browser trace cleanup reports {} canvases; expected {}",
            cleanup.canvas_count,
            canvas_ids.len()
        )
    }
    let required_attributes: Vec<String> = expected_attributes
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    if cleanup.canvas_attribute_names != required_attributes {
        bail!("browser trace cleanup does not record the required RITK attributes")
    }
    Ok(())
}
