//! Validation of pointer, wheel, and keyboard evidence in a browser trace.

use anyhow::{bail, Result};
use std::collections::{BTreeMap, BTreeSet};

use super::{cine_rate, KeyboardTraceKind, TraceAction, TraceInputMode};

/// Validate the required input actions for one trace mode.
pub(super) fn validate_actions(
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
