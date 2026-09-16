//! Validation of the RITK-owned meaning carried by a generic Métis canvas trace.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

mod cine_rate;
mod validation;

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
    if matches!(
        input_mode.keyboard_kind(),
        Some(KeyboardTraceKind::CineRate)
    ) {
        return cine_rate::validate_actions(actions, canvas_ids);
    }
    let actions_per_canvas = match input_mode {
        TraceInputMode::PointerWheel => 2,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation) => 3,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate) => {
            unreachable!("invariant: cine-rate actions return through their dedicated validator")
        }
    };
    let expected_action_count = canvas_ids.len() * actions_per_canvas;
    if actions.len() != expected_action_count {
        bail!(
            "browser trace contains {} actions; expected {expected_action_count}",
            actions.len()
        )
    }

    let mut counts: BTreeMap<String, (usize, usize, usize)> = canvas_ids
        .iter()
        .map(|id| (id.clone(), (0, 0, 0)))
        .collect();
    for action in actions {
        let Some((pointer_count, wheel_count, keyboard_count)) = counts.get_mut(&action.canvas)
        else {
            bail!(
                "browser trace action targets unknown canvas {:?}",
                action.canvas
            )
        };
        match action.action.as_str() {
            "trusted-pointer-drag" => *pointer_count += 1,
            "trusted-wheel" => *wheel_count += 1,
            "trusted-keyboard" => {
                let Some(kind) = input_mode.keyboard_kind() else {
                    bail!("browser trace contains keyboard evidence without keyboard validation")
                };
                validate_keyboard_action(action, kind)?;
                *keyboard_count += 1;
            }
            other => bail!("browser trace contains unsupported canvas action {other:?}"),
        }
    }
    for (id, (pointer_count, wheel_count, keyboard_count)) in counts {
        let valid = match input_mode {
            TraceInputMode::PointerWheel => pointer_count == 1 && wheel_count == 1,
            TraceInputMode::PointerWheelKeyboard(_) => {
                pointer_count == 1 && wheel_count == 1 && keyboard_count == 1
            }
        };
        if !valid {
            match input_mode {
                TraceInputMode::PointerWheel => {
                    bail!(
                        "canvas {id:?} requires one trusted pointer drag and one trusted wheel action"
                    )
                }
                TraceInputMode::PointerWheelKeyboard(_) => bail!(
                    "canvas {id:?} requires one trusted pointer drag, one trusted wheel action and one trusted keyboard action"
                ),
            }
        }
    }
    Ok(())
}

fn validate_keyboard_action(action: &TraceAction, kind: KeyboardTraceKind) -> Result<()> {
    let (expected_key, expected_code) = kind.key_code();
    if action.key.as_deref() != Some(expected_key)
        || action.code.as_deref() != Some(expected_code)
        || action.repeat != Some(false)
    {
        bail!(
            "canvas {:?} keyboard action has invalid key/code for {:?} and repeat=false",
            action.canvas,
            kind
        )
    }
    let Some(focus) = action.focus.as_ref() else {
        bail!(
            "canvas {:?} keyboard action is missing focus evidence",
            action.canvas
        )
    };
    if !focus.ok || focus.active_id.as_deref() != Some(action.canvas.as_str()) {
        bail!(
            "canvas {:?} keyboard action did not focus its target",
            action.canvas
        )
    }
    if action.observed_events.len() != 2 {
        bail!(
            "canvas {:?} keyboard action must contain exactly one keydown and one keyup event",
            action.canvas
        )
    }
    let mut phases = BTreeSet::new();
    for event in &action.observed_events {
        if !matches!(event.event_type.as_str(), "keydown" | "keyup") {
            bail!(
                "canvas {:?} keyboard evidence contains unsupported event {:?}",
                action.canvas,
                event.event_type
            )
        }
        if event.is_trusted != Some(true) {
            bail!(
                "canvas {:?} keyboard event {:?} was not trusted",
                action.canvas,
                event.event_type
            )
        }
        if event.target_id.as_deref() != Some(action.canvas.as_str()) {
            bail!(
                "canvas {:?} keyboard event {:?} targeted the wrong canvas",
                action.canvas,
                event.event_type
            )
        }
        if event.key.as_deref() != Some(expected_key)
            || event.code.as_deref() != Some(expected_code)
            || event.repeat != Some(false)
        {
            bail!(
                "canvas {:?} keyboard event {:?} has invalid key metadata",
                action.canvas,
                event.event_type
            )
        }
        for (name, value) in [
            ("alt", event.alt_key),
            ("ctrl", event.ctrl_key),
            ("meta", event.meta_key),
            ("shift", event.shift_key),
        ] {
            if value != Some(false) {
                bail!(
                    "canvas {:?} keyboard event {:?} has an active {name} modifier",
                    action.canvas,
                    event.event_type
                )
            }
        }
        if !phases.insert(event.event_type.as_str()) {
            bail!(
                "canvas {:?} keyboard evidence repeats {:?}",
                action.canvas,
                event.event_type
            )
        }
    }
    if phases != BTreeSet::from(["keydown", "keyup"]) {
        bail!(
            "canvas {:?} keyboard evidence must contain keydown and keyup",
            action.canvas
        )
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SnapshotPhase {
    Initial,
    AfterKeyboard,
    AfterRepeat,
    AfterDecrease,
    AfterDecreaseRepeat,
    AfterInput,
}

fn validate_snapshots(
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
) -> Result<()> {
    if matches!(
        input_mode.keyboard_kind(),
        Some(KeyboardTraceKind::CineRate)
    ) {
        return cine_rate::validate_snapshots(snapshots, canvas_ids);
    }
    let snapshots_per_canvas = match input_mode {
        TraceInputMode::PointerWheel => 2,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation) => 3,
        TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate) => {
            unreachable!("invariant: cine-rate snapshots return through their dedicated validator")
        }
    };
    let expected_snapshot_count = canvas_ids.len() * snapshots_per_canvas;
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
                } else if input_mode.keyboard_kind().is_some()
                    && snapshot.label == format!("{id}-after-keyboard")
                {
                    Some((axis, id.as_str(), SnapshotPhase::AfterKeyboard))
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
        validate_snapshot(
            snapshot,
            canvas_id,
            axis,
            phase,
            input_mode.expected_attributes(),
        )?;
    }

    let suffixes = match input_mode {
        TraceInputMode::PointerWheel => ["initial", "after-input"].as_slice(),
        TraceInputMode::PointerWheelKeyboard(_) => {
            ["initial", "after-keyboard", "after-input"].as_slice()
        }
    };
    for id in canvas_ids {
        for suffix in suffixes {
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
    expected_attributes: &[&str],
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
    if !snapshot.canvas.css_width.is_finite()
        || snapshot.canvas.css_width <= 0.0
        || snapshot.canvas.css_width > MAX_CANVAS_CSS_DIMENSION
        || !snapshot.canvas.css_height.is_finite()
        || snapshot.canvas.css_height <= 0.0
        || snapshot.canvas.css_height > MAX_CANVAS_CSS_DIMENSION
    {
        bail!("canvas {canvas_id:?} has invalid CSS dimensions")
    }
    if snapshot.canvas.attributes.len() != expected_attributes.len()
        || expected_attributes
            .iter()
            .any(|name| !snapshot.canvas.attributes.contains_key(*name))
    {
        bail!("canvas {canvas_id:?} does not carry the complete RITK attribute set")
    }

    let load_state = attribute(&snapshot.canvas, "data-ritk-load-state", canvas_id)?;
    let frame_state = attribute(&snapshot.canvas, "data-ritk-frame-state", canvas_id)?;
    let axis_value = attribute(&snapshot.canvas, "data-ritk-axis", canvas_id)?;
    let slice_index: u64 = parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-slice-index", canvas_id)?,
        "slice index",
        canvas_id,
    )?;
    let slice_count: u64 = parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-slice-count", canvas_id)?,
        "slice count",
        canvas_id,
    )?;
    let frame_width: u32 = parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-frame-width", canvas_id)?,
        "frame width",
        canvas_id,
    )?;
    let frame_height: u32 = parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-frame-height", canvas_id)?,
        "frame height",
        canvas_id,
    )?;
    if expected_attributes.contains(&"data-ritk-cine-fps") {
        let cine_fps: u32 = parse_attribute(
            attribute(&snapshot.canvas, "data-ritk-cine-fps", canvas_id)?,
            "cine FPS",
            canvas_id,
        )?;
        if !(1..=60).contains(&cine_fps) {
            bail!("canvas {canvas_id:?} reports an invalid cine FPS")
        }
        let frame_generation: u64 = parse_attribute(
            attribute(&snapshot.canvas, "data-ritk-frame-generation", canvas_id)?,
            "frame generation",
            canvas_id,
        )?;
        if frame_generation == 0 {
            bail!("canvas {canvas_id:?} reports a zero frame generation")
        }
    }

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
    if matches!(
        phase,
        SnapshotPhase::AfterKeyboard
            | SnapshotPhase::AfterRepeat
            | SnapshotPhase::AfterDecrease
            | SnapshotPhase::AfterDecreaseRepeat
            | SnapshotPhase::AfterInput
    ) && (load_state != "ready" || frame_state != "presented")
    {
        bail!("canvas {canvas_id:?} is not presented after trusted input")
    }
    Ok(())
}

fn validate_slice_progression(
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
    input_mode: TraceInputMode,
) -> Result<()> {
    for canvas_id in canvas_ids {
        let initial_label = format!("{canvas_id}-initial");
        let before_wheel_label = match input_mode {
            TraceInputMode::PointerWheel => initial_label.clone(),
            TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::Navigation) => {
                format!("{canvas_id}-after-keyboard")
            }
            TraceInputMode::PointerWheelKeyboard(KeyboardTraceKind::CineRate) => {
                format!("{canvas_id}-after-decrease-repeat")
            }
        };
        let after_label = format!("{canvas_id}-after-input");
        let initial = snapshots
            .iter()
            .find(|snapshot| snapshot.label == initial_label)
            .with_context(|| format!("browser trace is missing snapshot {initial_label:?}"))?;
        let before_wheel = snapshots
            .iter()
            .find(|snapshot| snapshot.label == before_wheel_label)
            .with_context(|| format!("browser trace is missing snapshot {before_wheel_label:?}"))?;
        let after = snapshots
            .iter()
            .find(|snapshot| snapshot.label == after_label)
            .with_context(|| format!("browser trace is missing snapshot {after_label:?}"))?;
        let initial_count: u64 = parse_attribute(
            attribute(&initial.canvas, "data-ritk-slice-count", canvas_id)?,
            "slice count",
            canvas_id,
        )?;
        let before_wheel_count: u64 = parse_attribute(
            attribute(&before_wheel.canvas, "data-ritk-slice-count", canvas_id)?,
            "slice count",
            canvas_id,
        )?;
        let after_count: u64 = parse_attribute(
            attribute(&after.canvas, "data-ritk-slice-count", canvas_id)?,
            "slice count",
            canvas_id,
        )?;
        if initial_count != before_wheel_count {
            bail!(
                "canvas {canvas_id:?} changed its slice count from {initial_count} to {before_wheel_count} before the wheel"
            )
        }
        if before_wheel_count != after_count {
            bail!(
                "canvas {canvas_id:?} changed its slice count from {before_wheel_count} to {after_count} after the wheel"
            )
        }
        if initial_count <= 1 {
            continue;
        }
        let before_wheel_index: u64 = parse_attribute(
            attribute(&before_wheel.canvas, "data-ritk-slice-index", canvas_id)?,
            "slice index",
            canvas_id,
        )?;
        let after_index: u64 = parse_attribute(
            attribute(&after.canvas, "data-ritk-slice-index", canvas_id)?,
            "slice index",
            canvas_id,
        )?;
        if before_wheel_index == after_index {
            bail!(
                "canvas {canvas_id:?} did not advance its multi-slice index after trusted wheel input"
            )
        }
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

fn parse_attribute<T>(value: &str, field: &str, canvas_id: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value
        .parse::<T>()
        .with_context(|| format!("canvas {canvas_id:?} has an invalid {field} value {value:?}"))
}

#[cfg(test)]
mod tests;
