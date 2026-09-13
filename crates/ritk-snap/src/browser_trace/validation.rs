//! Validation for browser-trace screenshots and teardown evidence.

use anyhow::{bail, Result};
use std::collections::BTreeSet;

use super::{TraceCleanup, TraceScreenshot, EXPECTED_ATTRIBUTES};

/// Validate the window and canvas screenshot manifest.
pub(super) fn validate_screenshots(
    screenshots: &[TraceScreenshot],
    canvas_ids: &[String],
) -> Result<()> {
    let expected_screenshot_count = 2 + canvas_ids.len() * 2;
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
                screenshot.label == format!("{id}-initial")
                    || screenshot.label == format!("{id}-after-input")
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
    }

    let mut expected_labels =
        BTreeSet::from(["window-initial".to_owned(), "window-final".to_owned()]);
    for id in canvas_ids {
        expected_labels.insert(format!("{id}-initial"));
        expected_labels.insert(format!("{id}-after-input"));
    }
    if labels != expected_labels {
        bail!(
            "browser trace screenshot labels do not cover the required window and canvas captures"
        )
    }
    Ok(())
}

/// Validate the input-release and canvas-attribute teardown evidence.
pub(super) fn validate_cleanup(cleanup: &TraceCleanup, canvas_ids: &[String]) -> Result<()> {
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
    let expected_attributes: Vec<String> = EXPECTED_ATTRIBUTES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    if cleanup.canvas_attribute_names != expected_attributes {
        bail!("browser trace cleanup does not record the required RITK attributes")
    }
    Ok(())
}
