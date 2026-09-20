//! Validation of canvas snapshots, attributes, and slice progression.

use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;
use std::str::FromStr;

use super::{
    cine_rate, KeyboardTraceKind, TraceCanvas, TraceInputMode, TraceSnapshot, CURSOR_ATTRIBUTES,
    INTERACTION_ATTRIBUTES, MAX_CANVAS_CSS_DIMENSION, MAX_CANVAS_DIMENSION,
    WINDOW_LEVEL_ATTRIBUTES,
};

#[derive(Clone, Copy)]
pub(super) enum SnapshotPhase {
    Initial,
    AfterKeyboard,
    AfterRepeat,
    AfterDecrease,
    AfterDecreaseRepeat,
    AfterInput,
}

pub(super) fn validate_snapshots(
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

pub(super) fn validate_snapshot(
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
    let attribute_names = snapshot
        .canvas
        .attributes
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    if !attribute_names_match(&attribute_names, expected_attributes) {
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
    if has_attribute_group(&snapshot.canvas, &WINDOW_LEVEL_ATTRIBUTES) {
        validate_window_level_attributes(snapshot, canvas_id)?;
    }
    if has_attribute_group(&snapshot.canvas, &INTERACTION_ATTRIBUTES) {
        validate_interaction_attributes(snapshot, canvas_id)?;
    }
    if has_attribute_group(&snapshot.canvas, &CURSOR_ATTRIBUTES) {
        validate_cursor_attributes(snapshot, canvas_id)?;
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

pub(super) fn attribute_names_match(actual: &[String], expected: &[&str]) -> bool {
    let actual = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let required = expected.iter().copied().collect::<BTreeSet<_>>();
    if !actual.is_superset(&required) {
        return false;
    }
    let extension = actual
        .difference(&required)
        .copied()
        .collect::<BTreeSet<_>>();
    [
        &WINDOW_LEVEL_ATTRIBUTES[..],
        &INTERACTION_ATTRIBUTES[..],
        &CURSOR_ATTRIBUTES[..],
    ]
    .into_iter()
    .all(|group| {
        let group = group.iter().copied().collect::<BTreeSet<_>>();
        extension.is_disjoint(&group) || extension.is_superset(&group)
    }) && extension.is_subset(
        &WINDOW_LEVEL_ATTRIBUTES
            .iter()
            .chain(INTERACTION_ATTRIBUTES.iter())
            .chain(CURSOR_ATTRIBUTES.iter())
            .copied()
            .collect::<BTreeSet<_>>(),
    )
}

fn has_attribute_group(canvas: &TraceCanvas, group: &[&str]) -> bool {
    group
        .iter()
        .any(|name| canvas.attributes.contains_key(*name))
}

fn validate_interaction_attributes(snapshot: &TraceSnapshot, canvas_id: &str) -> Result<()> {
    let cine_enabled = attribute(&snapshot.canvas, INTERACTION_ATTRIBUTES[0], canvas_id)?;
    if !matches!(cine_enabled, "true" | "false") {
        bail!("canvas {canvas_id:?} has an invalid cine-enabled value {cine_enabled:?}")
    }
    let _: u64 = parse_attribute(
        attribute(&snapshot.canvas, INTERACTION_ATTRIBUTES[1], canvas_id)?,
        "active tool index",
        canvas_id,
    )?;
    attribute(&snapshot.canvas, INTERACTION_ATTRIBUTES[2], canvas_id)?;
    Ok(())
}

fn validate_cursor_attributes(snapshot: &TraceSnapshot, canvas_id: &str) -> Result<()> {
    let visible = attribute(&snapshot.canvas, "data-ritk-crosshair-visible", canvas_id)?;
    if !matches!(visible, "true" | "false") {
        bail!("canvas {canvas_id:?} has an invalid crosshair visibility value {visible:?}")
    }
    let linked = snapshot
        .canvas
        .attributes
        .get("data-ritk-linked-cursor")
        .and_then(Option::as_deref)
        .unwrap_or("");
    if !linked.is_empty() {
        let values = linked.split(',').collect::<Vec<_>>();
        if values.len() != 3 {
            bail!("canvas {canvas_id:?} has an invalid linked cursor value {linked:?}")
        }
        for value in values {
            let _: u64 = parse_attribute(value, "linked cursor coordinate", canvas_id)?;
        }
    }
    for name in ["data-ritk-view-flip-h", "data-ritk-view-flip-v"] {
        let value = attribute(&snapshot.canvas, name, canvas_id)?;
        if !matches!(value, "true" | "false") {
            bail!("canvas {canvas_id:?} has an invalid orientation value {value:?}")
        }
    }
    let rotation = attribute(&snapshot.canvas, "data-ritk-view-rotation", canvas_id)?;
    if !matches!(rotation, "0" | "90" | "180" | "270") {
        bail!("canvas {canvas_id:?} has an invalid view rotation {rotation:?}")
    }
    Ok(())
}

fn validate_window_level_attributes(snapshot: &TraceSnapshot, canvas_id: &str) -> Result<()> {
    let center: f64 = parse_attribute(
        attribute(&snapshot.canvas, WINDOW_LEVEL_ATTRIBUTES[0], canvas_id)?,
        "window center",
        canvas_id,
    )?;
    if !center.is_finite() {
        bail!("canvas {canvas_id:?} reports a non-finite window center")
    }
    let width: f64 = parse_attribute(
        attribute(&snapshot.canvas, WINDOW_LEVEL_ATTRIBUTES[1], canvas_id)?,
        "window width",
        canvas_id,
    )?;
    if !width.is_finite() || width <= 0.0 {
        bail!("canvas {canvas_id:?} reports a non-positive or non-finite window width")
    }
    let Some(raw_index) = snapshot.canvas.attributes.get(WINDOW_LEVEL_ATTRIBUTES[2]) else {
        bail!("canvas {canvas_id:?} is missing window preset index attribute")
    };
    let Some(raw_index) = raw_index else {
        bail!("canvas {canvas_id:?} has a null window preset index")
    };
    if !raw_index.is_empty() {
        let _: u64 = parse_attribute(raw_index, "window preset index", canvas_id)?;
    }
    Ok(())
}

pub(super) fn validate_slice_progression(
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

pub(super) fn attribute<'a>(
    canvas: &'a TraceCanvas,
    name: &str,
    canvas_id: &str,
) -> Result<&'a str> {
    let Some(Some(value)) = canvas.attributes.get(name) else {
        bail!("canvas {canvas_id:?} is missing non-null attribute {name:?}")
    };
    if value.is_empty() {
        bail!("canvas {canvas_id:?} has an empty attribute {name:?}")
    }
    Ok(value)
}

pub(super) fn parse_attribute<T>(value: &str, field: &str, canvas_id: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value
        .parse::<T>()
        .with_context(|| format!("canvas {canvas_id:?} has an invalid {field} value {value:?}"))
}
