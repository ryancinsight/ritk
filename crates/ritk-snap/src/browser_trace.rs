//! Validation of the RITK-owned meaning carried by a generic Métis canvas trace.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

mod actions;
mod cine_rate;
mod snapshots;
mod validation;

use snapshots::SnapshotPhase;

const MAX_CANVAS_DIMENSION: u32 = 4_096;
const MAX_CANVAS_CSS_DIMENSION: f64 = 16_384.0;
const MAX_TRACE_BYTES: u64 = 512 * 1024;
const BASE_ATTRIBUTES: [&str; 7] = [
    "data-ritk-load-state",
    "data-ritk-frame-state",
    "data-ritk-axis",
    "data-ritk-slice-index",
    "data-ritk-slice-count",
    "data-ritk-frame-width",
    "data-ritk-frame-height",
];
const CINE_RATE_ATTRIBUTES: [&str; 10] = [
    "data-ritk-load-state",
    "data-ritk-frame-state",
    "data-ritk-axis",
    "data-ritk-slice-index",
    "data-ritk-slice-count",
    "data-ritk-frame-width",
    "data-ritk-frame-height",
    "data-ritk-cine-fps",
    "data-ritk-frame-generation",
    "data-ritk-display-aspect",
];
const WINDOW_LEVEL_ATTRIBUTES: [&str; 3] = [
    "data-ritk-window-center",
    "data-ritk-window-width",
    "data-ritk-window-preset-index",
];
const INTERACTION_ATTRIBUTES: [&str; 3] = [
    "data-ritk-cine-enabled",
    "data-ritk-active-tool-index",
    "data-ritk-active-tool",
];
const CURSOR_ATTRIBUTES: [&str; 5] = [
    "data-ritk-crosshair-visible",
    "data-ritk-linked-cursor",
    "data-ritk-view-flip-h",
    "data-ritk-view-flip-v",
    "data-ritk-view-rotation",
];
const DEFAULT_CANVAS_IDS: [&str; 3] =
    ["ritk-snap-axial", "ritk-snap-coronal", "ritk-snap-sagittal"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KeyboardTraceKind {
    /// The focused ArrowDown navigation profile.
    Navigation,
    /// The focused Equal cine-rate profile.
    CineRate,
}

impl KeyboardTraceKind {
    fn key_code(self) -> (&'static str, &'static str) {
        match self {
            Self::Navigation => ("ArrowDown", "ArrowDown"),
            Self::CineRate => ("=", "Equal"),
        }
    }
}

/// Input evidence required by a browser trace validation run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TraceInputMode {
    /// Require one trusted pointer drag and wheel action per canvas.
    PointerWheel,
    /// Also require a focused keyboard pair per canvas.
    PointerWheelKeyboard(KeyboardTraceKind),
}

impl TraceInputMode {
    fn keyboard_kind(self) -> Option<KeyboardTraceKind> {
        match self {
            Self::PointerWheel => None,
            Self::PointerWheelKeyboard(kind) => Some(kind),
        }
    }

    fn expected_attributes(self) -> &'static [&'static str] {
        match self.keyboard_kind() {
            Some(KeyboardTraceKind::CineRate) => &CINE_RATE_ATTRIBUTES,
            Some(KeyboardTraceKind::Navigation) | None => &BASE_ATTRIBUTES,
        }
    }
}

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
    #[serde(default)]
    metrics: Option<TraceMetrics>,
    cleanup: TraceCleanup,
}

#[derive(Debug, Deserialize)]
struct TraceMetrics {
    #[serde(default)]
    device_scale: Option<TraceDeviceScale>,
}

#[derive(Debug, Deserialize)]
struct TraceDeviceScale {
    device_pixel_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct TraceAction {
    action: String,
    canvas: String,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    repeat: Option<bool>,
    #[serde(default)]
    focus: Option<TraceFocus>,
    #[serde(default)]
    observed_events: Vec<TraceEvent>,
}

#[derive(Debug, Deserialize)]
struct TraceFocus {
    ok: bool,
    active_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TraceEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    is_trusted: Option<bool>,
    #[serde(default)]
    target_id: Option<String>,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    repeat: Option<bool>,
    #[serde(default)]
    alt_key: Option<bool>,
    #[serde(default)]
    ctrl_key: Option<bool>,
    #[serde(default)]
    meta_key: Option<bool>,
    #[serde(default)]
    shift_key: Option<bool>,
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
    css_width: f64,
    css_height: f64,
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
    input_mode: TraceInputMode,
) -> Result<BrowserTraceReport> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open browser trace {}", path.display()))?;
    let length = file
        .metadata()
        .with_context(|| format!("failed to inspect browser trace {}", path.display()))?
        .len();
    if length > MAX_TRACE_BYTES {
        bail!("browser trace is {length} bytes; the {MAX_TRACE_BYTES}-byte limit was exceeded")
    }
    let capacity = usize::try_from(length).context("browser trace length does not fit memory")?;
    let mut bytes = Vec::with_capacity(capacity.saturating_add(1));
    file.take(MAX_TRACE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read browser trace {}", path.display()))?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_TRACE_BYTES) {
        bail!("browser trace exceeded the {MAX_TRACE_BYTES}-byte limit while reading")
    }
    let document: TraceDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse browser trace {}", path.display()))?;
    let canvas_ids = resolve_canvas_ids(requested_canvas_ids)?;
    validate_document(&document, &canvas_ids, input_mode)
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
    input_mode: TraceInputMode,
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
    validate_actions(&document.actions, canvas_ids, input_mode)?;
    validate_snapshots(&document.snapshots, canvas_ids, input_mode)?;
    validate_slice_progression(&document.snapshots, canvas_ids, input_mode)?;
    if matches!(
        input_mode.keyboard_kind(),
        Some(KeyboardTraceKind::CineRate)
    ) {
        cine_rate::validate_snapshot_progression(&document.snapshots, canvas_ids)?;
    }
    let device_pixel_ratio = if matches!(
        input_mode.keyboard_kind(),
        Some(KeyboardTraceKind::CineRate)
    ) {
        let ratio = document
            .metrics
            .as_ref()
            .context("cine browser trace is missing device-scale metrics")?
            .device_scale
            .as_ref()
            .context("cine browser trace is missing device-scale metrics")?
            .device_pixel_ratio;
        if !ratio.is_finite() || !(0.5..=4.0).contains(&ratio) {
            bail!("cine browser trace reports an invalid device pixel ratio")
        }
        Some(ratio)
    } else {
        None
    };
    validation::validate_screenshots(
        &document.screenshots,
        &document.snapshots,
        canvas_ids,
        input_mode,
        device_pixel_ratio,
    )?;
    validation::validate_cleanup(
        &document.cleanup,
        canvas_ids,
        input_mode.expected_attributes(),
    )?;

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

fn validate_actions(
    actions: &[TraceAction],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
) -> Result<()> {
    actions::validate_actions(actions, canvas_ids, input_mode)
}

fn validate_snapshots(
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
) -> Result<()> {
    snapshots::validate_snapshots(snapshots, canvas_ids, input_mode)
}

fn validate_snapshot(
    snapshot: &TraceSnapshot,
    canvas_id: &str,
    axis: usize,
    phase: SnapshotPhase,
    expected_attributes: &[&str],
) -> Result<()> {
    snapshots::validate_snapshot(snapshot, canvas_id, axis, phase, expected_attributes)
}

fn validate_slice_progression(
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
) -> Result<()> {
    snapshots::validate_slice_progression(snapshots, canvas_ids, input_mode)
}

fn attribute_names_match(actual: &[String], expected: &[&str]) -> bool {
    snapshots::attribute_names_match(actual, expected)
}

fn attribute<'a>(canvas: &'a TraceCanvas, name: &str, canvas_id: &str) -> Result<&'a str> {
    snapshots::attribute(canvas, name, canvas_id)
}

fn parse_attribute<T>(value: &str, field: &str, canvas_id: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    snapshots::parse_attribute(value, field, canvas_id)
}
#[cfg(test)]
mod tests;
