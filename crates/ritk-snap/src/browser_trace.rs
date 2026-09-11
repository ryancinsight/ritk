//! Validation of the RITK-owned meaning carried by a generic Métis canvas trace.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

const MAX_CANVAS_DIMENSION: u32 = 4_096;
const MAX_TRACE_BYTES: u64 = 512 * 1024;
const EXPECTED_ATTRIBUTES: [&str; 7] = [
    "data-ritk-load-state",
    "data-ritk-frame-state",
    "data-ritk-axis",
    "data-ritk-slice-index",
    "data-ritk-slice-count",
    "data-ritk-frame-width",
    "data-ritk-frame-height",
];
const DEFAULT_CANVAS_IDS: [&str; 3] =
    ["ritk-snap-axial", "ritk-snap-coronal", "ritk-snap-sagittal"];

#[derive(Debug, Deserialize)]
struct TraceDocument {
    schema: u8,
    status: String,
    engine: String,
    bridge: String,
    revision: String,
    #[serde(default)]
    consumer_revision: Option<String>,
    actions: Vec<TraceAction>,
    snapshots: Vec<TraceSnapshot>,
    screenshots: Vec<TraceScreenshot>,
    cleanup: TraceCleanup,
}

#[derive(Debug, Deserialize)]
struct TraceAction {
    action: String,
    canvas: String,
}

#[derive(Debug, Deserialize)]
struct TraceSnapshot {
    label: String,
    canvas: TraceCanvas,
}

#[derive(Debug, Deserialize)]
struct TraceCanvas {
    id: String,
    width: u32,
    height: u32,
    attributes: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Deserialize)]
struct TraceScreenshot {
    label: String,
    #[serde(default)]
    scope: Option<String>,
    width: u32,
    height: u32,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct TraceCleanup {
    active_input_sources_released: bool,
    canvas_count: usize,
    canvas_attribute_names: Vec<String>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct BrowserTraceReport {
    engine: String,
    revision: String,
    consumer_revision: String,
    canvas_count: usize,
}

impl fmt::Display for BrowserTraceReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "validated RITK browser trace: engine={}, canvases={}, metis_revision={}, ritk_revision={}",
            self.engine, self.canvas_count, self.revision, self.consumer_revision
        )
    }
}

/// Read and validate a schema-1 browser trace produced by the generic runner.
pub(crate) fn validate_file(
    path: &Path,
    requested_canvas_ids: &[String],
) -> Result<BrowserTraceReport> {
    let length = fs::metadata(path)
        .with_context(|| format!("failed to stat browser trace {}", path.display()))?
        .len();
    if length > MAX_TRACE_BYTES {
        bail!("browser trace is {length} bytes; the {MAX_TRACE_BYTES}-byte limit was exceeded")
    }
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read browser trace {}", path.display()))?;
    let document: TraceDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse browser trace {}", path.display()))?;
    let canvas_ids = resolve_canvas_ids(requested_canvas_ids)?;
    validate_document(&document, &canvas_ids)
}

fn resolve_canvas_ids(requested: &[String]) -> Result<Vec<String>> {
    let ids = if requested.is_empty() {
        DEFAULT_CANVAS_IDS
            .iter()
            .map(|id| (*id).to_owned())
            .collect()
    } else {
        if requested.len() != DEFAULT_CANVAS_IDS.len() {
            bail!("RITK browser trace requires exactly three canvas identifiers")
        }
        requested.to_vec()
    };

    let mut seen = BTreeSet::new();
    for id in &ids {
        if !is_canvas_id(id) {
            bail!("canvas identifier is not a bounded HTML id: {id:?}")
        }
        if !seen.insert(id) {
            bail!("canvas identifier is repeated: {id:?}")
        }
    }
    Ok(ids)
}

fn is_canvas_id(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && value.len() <= 128
        && characters
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn validate_document(
    document: &TraceDocument,
    canvas_ids: &[String],
) -> Result<BrowserTraceReport> {
    if document.schema != 1 {
        bail!(
            "unsupported browser trace schema {}; expected 1",
            document.schema
        )
    }
    if document.status != "passed" {
        bail!(
            "browser trace status is {:?}; expected passed",
            document.status
        )
    }
    if document.bridge != "canvas" {
        bail!(
            "browser trace bridge is {:?}; expected canvas",
            document.bridge
        )
    }
    if !matches!(document.engine.as_str(), "chromium" | "firefox" | "webkit") {
        bail!(
            "browser trace engine {:?} is outside the supported matrix",
            document.engine
        )
    }
    validate_revision("Metis", &document.revision)?;
    let consumer_revision = document
        .consumer_revision
        .as_deref()
        .context("browser trace is missing the RITK consumer revision")?;
    validate_revision("RITK", consumer_revision)?;
    validate_actions(&document.actions, canvas_ids)?;
    validate_snapshots(&document.snapshots, canvas_ids)?;
    validate_screenshots(&document.screenshots, canvas_ids)?;
    validate_cleanup(&document.cleanup, canvas_ids)?;

    Ok(BrowserTraceReport {
        engine: document.engine.clone(),
        revision: document.revision.clone(),
        consumer_revision: consumer_revision.to_owned(),
        canvas_count: canvas_ids.len(),
    })
}

fn validate_revision(name: &str, revision: &str) -> Result<()> {
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} revision must be a 40-hex Git revision")
    }
    Ok(())
}

fn validate_actions(actions: &[TraceAction], canvas_ids: &[String]) -> Result<()> {
    let expected_action_count = canvas_ids.len() * 2;
    if actions.len() != expected_action_count {
        bail!(
            "browser trace contains {} actions; expected {expected_action_count}",
            actions.len()
        )
    }

    let mut counts: BTreeMap<String, (usize, usize)> =
        canvas_ids.iter().map(|id| (id.clone(), (0, 0))).collect();
    for action in actions {
        let Some((pointer_count, wheel_count)) = counts.get_mut(&action.canvas) else {
            bail!(
                "browser trace action targets unknown canvas {:?}",
                action.canvas
            )
        };
        match action.action.as_str() {
            "trusted-pointer-drag" => *pointer_count += 1,
            "trusted-wheel" => *wheel_count += 1,
            other => bail!("browser trace contains unsupported canvas action {other:?}"),
        }
    }
    for (id, (pointer_count, wheel_count)) in counts {
        if pointer_count != 1 || wheel_count != 1 {
            bail!("canvas {id:?} requires one trusted pointer drag and one trusted wheel action")
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SnapshotPhase {
    Initial,
    AfterInput,
}

fn validate_snapshots(snapshots: &[TraceSnapshot], canvas_ids: &[String]) -> Result<()> {
    let expected_snapshot_count = canvas_ids.len() * 2;
    if snapshots.len() != expected_snapshot_count {
        bail!(
            "browser trace contains {} snapshots; expected {expected_snapshot_count}",
            snapshots.len()
        )
    }

    let mut labels = BTreeSet::new();
    for snapshot in snapshots {
        let Some((axis, canvas_id, phase)) =
            canvas_ids.iter().enumerate().find_map(|(axis, id)| {
                if snapshot.label == format!("{id}-initial") {
                    Some((axis, id.as_str(), SnapshotPhase::Initial))
                } else if snapshot.label == format!("{id}-after-input") {
                    Some((axis, id.as_str(), SnapshotPhase::AfterInput))
                } else {
                    None
                }
            })
        else {
            bail!(
                "browser trace contains unknown or malformed snapshot label {:?}",
                snapshot.label
            )
        };
        if !labels.insert(snapshot.label.clone()) {
            bail!("browser trace repeats snapshot label {:?}", snapshot.label)
        }
        validate_snapshot(snapshot, canvas_id, axis, phase)?;
    }

    for id in canvas_ids {
        for suffix in ["initial", "after-input"] {
            let label = format!("{id}-{suffix}");
            if !labels.contains(&label) {
                bail!("browser trace is missing snapshot {label:?}")
            }
        }
    }
    Ok(())
}

fn validate_snapshot(
    snapshot: &TraceSnapshot,
    canvas_id: &str,
    axis: usize,
    phase: SnapshotPhase,
) -> Result<()> {
    if snapshot.canvas.id != canvas_id {
        bail!(
            "snapshot {:?} identifies canvas {:?}; expected {:?}",
            snapshot.label,
            snapshot.canvas.id,
            canvas_id
        )
    }
    if snapshot.canvas.width == 0 || snapshot.canvas.width > MAX_CANVAS_DIMENSION {
        bail!(
            "canvas {canvas_id:?} has invalid intrinsic width {}",
            snapshot.canvas.width
        )
    }
    if snapshot.canvas.height == 0 || snapshot.canvas.height > MAX_CANVAS_DIMENSION {
        bail!(
            "canvas {canvas_id:?} has invalid intrinsic height {}",
            snapshot.canvas.height
        )
    }
    if snapshot.canvas.attributes.len() != EXPECTED_ATTRIBUTES.len()
        || EXPECTED_ATTRIBUTES
            .iter()
            .any(|name| !snapshot.canvas.attributes.contains_key(*name))
    {
        bail!("canvas {canvas_id:?} does not carry the complete RITK attribute set")
    }

    let load_state = attribute(&snapshot.canvas, "data-ritk-load-state", canvas_id)?;
    let frame_state = attribute(&snapshot.canvas, "data-ritk-frame-state", canvas_id)?;
    let axis_value = attribute(&snapshot.canvas, "data-ritk-axis", canvas_id)?;
    let slice_index = parse_u64(
        attribute(&snapshot.canvas, "data-ritk-slice-index", canvas_id)?,
        "slice index",
        canvas_id,
    )?;
    let slice_count = parse_u64(
        attribute(&snapshot.canvas, "data-ritk-slice-count", canvas_id)?,
        "slice count",
        canvas_id,
    )?;
    let frame_width = parse_u32(
        attribute(&snapshot.canvas, "data-ritk-frame-width", canvas_id)?,
        "frame width",
        canvas_id,
    )?;
    let frame_height = parse_u32(
        attribute(&snapshot.canvas, "data-ritk-frame-height", canvas_id)?,
        "frame height",
        canvas_id,
    )?;

    if !matches!(load_state, "empty" | "ready") {
        bail!("canvas {canvas_id:?} has invalid load state {load_state:?}")
    }
    if !matches!(frame_state, "empty" | "presented") {
        bail!("canvas {canvas_id:?} has invalid frame state {frame_state:?}")
    }
    if axis_value != axis.to_string() {
        bail!("canvas {canvas_id:?} reports axis {axis_value:?}; expected {axis}")
    }
    if slice_count == 0 || slice_index >= slice_count {
        bail!("canvas {canvas_id:?} reports an invalid slice range")
    }
    if frame_state == "presented" {
        if load_state != "ready"
            || frame_width == 0
            || frame_height == 0
            || frame_width != snapshot.canvas.width
            || frame_height != snapshot.canvas.height
        {
            bail!("canvas {canvas_id:?} has inconsistent presented frame dimensions")
        }
    } else if frame_width != 0 || frame_height != 0 {
        bail!("canvas {canvas_id:?} has dimensions for an empty frame")
    }
    if matches!(phase, SnapshotPhase::AfterInput)
        && (load_state != "ready" || frame_state != "presented")
    {
        bail!("canvas {canvas_id:?} is not presented after trusted input")
    }
    Ok(())
}

fn attribute<'a>(canvas: &'a TraceCanvas, name: &str, canvas_id: &str) -> Result<&'a str> {
    let Some(Some(value)) = canvas.attributes.get(name) else {
        bail!("canvas {canvas_id:?} is missing non-null attribute {name:?}")
    };
    if value.is_empty() {
        bail!("canvas {canvas_id:?} has an empty attribute {name:?}")
    }
    Ok(value)
}

fn parse_u64(value: &str, field: &str, canvas_id: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .with_context(|| format!("canvas {canvas_id:?} has an invalid {field} value {value:?}"))
}

fn parse_u32(value: &str, field: &str, canvas_id: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .with_context(|| format!("canvas {canvas_id:?} has an invalid {field} value {value:?}"))
}

fn validate_screenshots(screenshots: &[TraceScreenshot], canvas_ids: &[String]) -> Result<()> {
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

fn validate_cleanup(cleanup: &TraceCleanup, canvas_ids: &[String]) -> Result<()> {
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

#[cfg(test)]
mod tests;
