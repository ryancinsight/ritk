//! Validation for the cine-rate input and presentation sequence.

use anyhow::{bail, Context, Result};

use super::{
    attribute, parse_attribute, validate_snapshot, SnapshotPhase, TraceAction, TraceEvent,
    TraceSnapshot, CINE_RATE_ATTRIBUTES,
};

pub(super) const SNAPSHOT_SUFFIXES: [&str; 6] = [
    "initial",
    "after-keyboard",
    "after-repeat",
    "after-decrease",
    "after-decrease-repeat",
    "after-input",
];

const ACTIONS_PER_CANVAS: usize = 6;

#[derive(Clone, Copy)]
struct KeyboardAction<'a> {
    key: &'a str,
    code: &'a str,
    repeat: bool,
}

/// Validate the exact rate-control, pointer, and wheel action sequence.
pub(super) fn validate_actions(actions: &[TraceAction], canvas_ids: &[String]) -> Result<()> {
    let expected_action_count = canvas_ids.len() * ACTIONS_PER_CANVAS;
    if actions.len() != expected_action_count {
        bail!(
            "browser trace contains {} actions; expected {expected_action_count}",
            actions.len()
        )
    }

    for (canvas_id, canvas_actions) in canvas_ids
        .iter()
        .zip(actions.chunks_exact(ACTIONS_PER_CANVAS))
    {
        let [increase, repeat_increase, decrease, repeat_decrease, pointer, wheel] = canvas_actions
        else {
            unreachable!("invariant: chunks_exact yields six-action cine slices")
        };
        for (action, expected) in [increase, repeat_increase, decrease, repeat_decrease]
            .into_iter()
            .zip([
                KeyboardAction {
                    key: "=",
                    code: "Equal",
                    repeat: false,
                },
                KeyboardAction {
                    key: "=",
                    code: "Equal",
                    repeat: true,
                },
                KeyboardAction {
                    key: "-",
                    code: "Minus",
                    repeat: false,
                },
                KeyboardAction {
                    key: "-",
                    code: "Minus",
                    repeat: true,
                },
            ])
        {
            validate_keyboard_action(action, canvas_id, expected)?;
        }
        validate_input_action(pointer, canvas_id, "trusted-pointer-drag")?;
        validate_input_action(wheel, canvas_id, "trusted-wheel")?;
    }
    Ok(())
}

fn validate_input_action(action: &TraceAction, canvas_id: &str, expected: &str) -> Result<()> {
    if action.canvas != canvas_id || action.action != expected {
        bail!(
            "canvas {canvas_id:?} cine action sequence expected {expected:?}, found {:?} targeting {:?}",
            action.action,
            action.canvas
        )
    }
    Ok(())
}

fn validate_keyboard_action(
    action: &TraceAction,
    canvas_id: &str,
    expected: KeyboardAction<'_>,
) -> Result<()> {
    if action.canvas != canvas_id || action.action != "trusted-keyboard" {
        bail!(
            "canvas {canvas_id:?} cine action sequence expected trusted-keyboard, found {:?} targeting {:?}",
            action.action,
            action.canvas
        )
    }
    if action.key.as_deref() != Some(expected.key)
        || action.code.as_deref() != Some(expected.code)
        || action.repeat != Some(expected.repeat)
    {
        bail!("canvas {canvas_id:?} cine keyboard action has invalid key/code/repeat metadata")
    }
    let Some(focus) = action.focus.as_ref() else {
        bail!("canvas {canvas_id:?} cine keyboard action is missing focus evidence")
    };
    if !focus.ok || focus.active_id.as_deref() != Some(canvas_id) {
        bail!("canvas {canvas_id:?} cine keyboard action did not focus its target")
    }

    if expected.repeat {
        let [keydown, keyup] = action.observed_events.as_slice() else {
            bail!(
                "canvas {canvas_id:?} repeated cine action must observe repeated keydown followed by non-repeated keyup"
            )
        };
        validate_event(
            keydown,
            canvas_id,
            expected.key,
            expected.code,
            "keydown",
            true,
        )?;
        validate_event(
            keyup,
            canvas_id,
            expected.key,
            expected.code,
            "keyup",
            false,
        )?;
    } else {
        let [keydown] = action.observed_events.as_slice() else {
            bail!("canvas {canvas_id:?} non-repeated cine action must observe one keydown")
        };
        validate_event(
            keydown,
            canvas_id,
            expected.key,
            expected.code,
            "keydown",
            false,
        )?;
    }
    Ok(())
}

fn validate_event(
    event: &TraceEvent,
    canvas_id: &str,
    key: &str,
    code: &str,
    event_type: &str,
    repeat: bool,
) -> Result<()> {
    if event.event_type != event_type
        || event.is_trusted != Some(true)
        || event.target_id.as_deref() != Some(canvas_id)
        || event.key.as_deref() != Some(key)
        || event.code.as_deref() != Some(code)
        || event.repeat != Some(repeat)
    {
        bail!(
            "canvas {canvas_id:?} cine {event_type} event has invalid trusted target/key/code/repeat metadata"
        )
    }
    for (name, value) in [
        ("alt", event.alt_key),
        ("ctrl", event.ctrl_key),
        ("meta", event.meta_key),
        ("shift", event.shift_key),
    ] {
        if value != Some(false) {
            bail!("canvas {canvas_id:?} cine {event_type} event has an active {name} modifier")
        }
    }
    Ok(())
}

/// Validate the exact snapshot labels and each stage's canvas state.
pub(super) fn validate_snapshots(snapshots: &[TraceSnapshot], canvas_ids: &[String]) -> Result<()> {
    let expected_snapshot_count = canvas_ids.len() * SNAPSHOT_SUFFIXES.len();
    if snapshots.len() != expected_snapshot_count {
        bail!(
            "browser trace contains {} snapshots; expected {expected_snapshot_count}",
            snapshots.len()
        )
    }

    for (axis, (canvas_id, canvas_snapshots)) in canvas_ids
        .iter()
        .zip(snapshots.chunks_exact(SNAPSHOT_SUFFIXES.len()))
        .enumerate()
    {
        for ((snapshot, suffix), phase) in canvas_snapshots.iter().zip(SNAPSHOT_SUFFIXES).zip([
            SnapshotPhase::Initial,
            SnapshotPhase::AfterKeyboard,
            SnapshotPhase::AfterRepeat,
            SnapshotPhase::AfterDecrease,
            SnapshotPhase::AfterDecreaseRepeat,
            SnapshotPhase::AfterInput,
        ]) {
            let expected_label = format!("{canvas_id}-{suffix}");
            if snapshot.label != expected_label {
                bail!(
                    "canvas {canvas_id:?} cine snapshot sequence expected {expected_label:?}, found {:?}",
                    snapshot.label
                )
            }
            validate_snapshot(snapshot, canvas_id, axis, phase, &CINE_RATE_ATTRIBUTES)?;
        }
    }
    Ok(())
}

/// Validate rate, render-generation, geometry, and slice invariants between stages.
pub(super) fn validate_snapshot_progression(
    snapshots: &[TraceSnapshot],
    canvas_ids: &[String],
) -> Result<()> {
    for canvas_id in canvas_ids {
        let [initial, after_increase, after_repeat_increase, after_decrease, after_repeat_decrease, after_input] =
            SNAPSHOT_SUFFIXES.map(|suffix| snapshot_for_stage(snapshots, canvas_id, suffix));
        let stages = [
            initial?,
            after_increase?,
            after_repeat_increase?,
            after_decrease?,
            after_repeat_decrease?,
            after_input?,
        ];
        let [initial, after_increase, after_repeat_increase, after_decrease, after_repeat_decrease, after_input] =
            stages;

        let [initial_rate, after_increase_rate, after_repeat_increase_rate, after_decrease_rate, after_repeat_decrease_rate, after_input_rate] = [
            cine_rate(initial, canvas_id)?,
            cine_rate(after_increase, canvas_id)?,
            cine_rate(after_repeat_increase, canvas_id)?,
            cine_rate(after_decrease, canvas_id)?,
            cine_rate(after_repeat_decrease, canvas_id)?,
            cine_rate(after_input, canvas_id)?,
        ];
        if !(2..=59).contains(&initial_rate) {
            bail!("canvas {canvas_id:?} initial cine FPS must be an integer from 2 through 59")
        }
        let increased_rate = initial_rate + 1;
        if [
            after_increase_rate,
            after_repeat_increase_rate,
            after_decrease_rate,
            after_repeat_decrease_rate,
            after_input_rate,
        ] != [
            increased_rate,
            increased_rate,
            initial_rate,
            initial_rate,
            initial_rate,
        ] {
            bail!(
                "canvas {canvas_id:?} cine FPS must change by exactly +1 and -1 while repeats and pointer/wheel input preserve the rate"
            )
        }

        let [initial_generation, after_increase_generation, after_repeat_increase_generation, after_decrease_generation, after_repeat_decrease_generation, after_input_generation] = [
            frame_generation(initial, canvas_id)?,
            frame_generation(after_increase, canvas_id)?,
            frame_generation(after_repeat_increase, canvas_id)?,
            frame_generation(after_decrease, canvas_id)?,
            frame_generation(after_repeat_decrease, canvas_id)?,
            frame_generation(after_input, canvas_id)?,
        ];
        let expected_after_increase = initial_generation
            .checked_add(1)
            .context("cine frame generation overflowed after Equal input")?;
        let expected_after_decrease = expected_after_increase
            .checked_add(1)
            .context("cine frame generation overflowed after Minus input")?;
        if [
            after_increase_generation,
            after_repeat_increase_generation,
            after_decrease_generation,
            after_repeat_decrease_generation,
        ] != [
            expected_after_increase,
            expected_after_increase,
            expected_after_decrease,
            expected_after_decrease,
        ] {
            bail!(
                "canvas {canvas_id:?} frame generation must advance once for effective rate actions and remain equal for repeats"
            )
        }
        if after_input_generation <= expected_after_decrease {
            bail!("canvas {canvas_id:?} did not render a fresh frame after pointer and wheel input")
        }

        let initial_slice_index = attribute(&initial.canvas, "data-ritk-slice-index", canvas_id)?;
        let initial_slice_count = attribute(&initial.canvas, "data-ritk-slice-count", canvas_id)?;
        for snapshot in [
            after_increase,
            after_repeat_increase,
            after_decrease,
            after_repeat_decrease,
        ] {
            if attribute(&snapshot.canvas, "data-ritk-slice-index", canvas_id)?
                != initial_slice_index
                || attribute(&snapshot.canvas, "data-ritk-slice-count", canvas_id)?
                    != initial_slice_count
            {
                bail!("canvas {canvas_id:?} changed its slice during a cine-rate action")
            }
            if snapshot.canvas.width != initial.canvas.width
                || snapshot.canvas.height != initial.canvas.height
            {
                bail!("canvas {canvas_id:?} changed intrinsic dimensions during a cine-rate action")
            }
        }
    }
    Ok(())
}

fn cine_rate(snapshot: &TraceSnapshot, canvas_id: &str) -> Result<u32> {
    parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-cine-fps", canvas_id)?,
        "cine FPS",
        canvas_id,
    )
}

fn frame_generation(snapshot: &TraceSnapshot, canvas_id: &str) -> Result<u64> {
    parse_attribute(
        attribute(&snapshot.canvas, "data-ritk-frame-generation", canvas_id)?,
        "frame generation",
        canvas_id,
    )
}

fn snapshot_for_stage<'a>(
    snapshots: &'a [TraceSnapshot],
    canvas_id: &str,
    suffix: &str,
) -> Result<&'a TraceSnapshot> {
    let label = format!("{canvas_id}-{suffix}");
    snapshots
        .iter()
        .find(|snapshot| snapshot.label == label)
        .with_context(|| format!("browser trace is missing snapshot {label:?}"))
}
