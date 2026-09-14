# ADR 0031: Bounded browser canvas input adapter

Status: Accepted

Date: 2026-09-11

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

## Context

The RITK browser entrypoints already receive bounded file bytes from Métis and
present RITK-owned frames through its canvas seam. Pointer and wheel input was
still absent, so the browser path could not exercise the shared presentation
reducer or the same cancellation behavior as the native Métis host.

The browser provider reports target-local coordinates, browser wheel units and
bounded keyboard metadata; RITK's action contract requires finite coordinates,
one declared wheel unit, shared virtual-key values and a viewport mapping for
each displayed frame. Three orthogonal canvases also need an explicit axis
route. DICOM parsing, metadata, geometry and viewer state must remain inside
RITK.

## Decision

`WebCanvasPresenter::from_canvas_id_with_input` retains Métis's bounded input
listeners. `take_events` translates pointer, wheel and keyboard records into
`PresentationEvent`, normalizes line and page deltas to CSS-pixel host units,
maps browser key codes to the shared virtual-key contract, preserves key-down
repeat state, rejects unsupported or non-finite values, and ignores non-primary
touch pointers because the current RITK dispatcher has one primary gesture
state. Unknown browser keys do not create presentation events.

The browser viewer drains each canvas queue once per browser animation frame.
It constructs a viewport from the current RITK frame dimensions, routes each
orthogonal canvas to its axis, applies the existing `SnapApp` action reducer,
and invalidates the frame
only when the reducer requests repaint, and cancels the gesture on translation
or reducer failure. A browser pointer cancel becomes the existing typed
`PointerCancelled` viewer action.

Revision 2026-09-11: the browser loop releases the Métis mount when a
presenter, input, or timer error ends the task, before the task exits. This
keeps listener and pointer-capture ownership generation-scoped and makes a
subsequent route remount independent of the failed task.

Revision 2026-09-13: Moirai's `WebAnimationFrame` future schedules the browser
loop at `requestAnimationFrame` and cancels its registration when dropped. The
loop releases the Métis mount when a presenter, input, or animation-frame error
ends the task, before the task exits. This keeps listener and pointer-capture
ownership generation-scoped and makes a subsequent route remount independent
of the failed task.

Revision 2026-09-14: the adapter consumes Métis keyboard down/up records. It
maps physical browser codes for cine, navigation and tool shortcuts to the
existing RITK virtual-key values, preserves auto-repeat on key-down events and
ignores unsupported codes. The mapping is format-neutral; DICOM and viewer
semantics remain in the RITK reducer.

Revision 2026-09-14: each RITK canvas is assigned `tabindex="0"` at the
consumer boundary before its input listeners are used. Keyboard focus therefore
targets the canvas that owns the bounded event queue instead of depending on
browser markup defaults.

Revision 2026-09-14: hosted run
[34895454734](https://github.com/ryancinsight/ritk/actions/runs/34895454734)
validated the focused keyboard path on Chromium 152 and Firefox 155 while
rendering the saved 94-file MRI-DIR study. Safari 26.6.2 accepted the chooser
event but rejected the first bounded file read, so it has no canvas or keyboard
claim.

## Rejected alternative

Handling pointer state in Métis would duplicate RITK's viewer reducer and make
clinical interaction semantics part of the GUI host. Passing raw DOM events
would couple RITK to a browser runtime and lose the bounded provider contract.

## Verification

The native RITK action suite covers pointer-cancel reduction, browser key-code
mapping and the existing wheel, viewport and cancellation laws. The RITK WASM
target is checked against the merged Metis input seam; the browser canvas
consumer makes its focus contract explicit with `tabindex="0"`. The existing
packaged synthetic DICOM capture continues to prove byte-to-frame ownership.
Physical pointer, keyboard-driver, cross-engine, GPU and full-window captures
remain open evidence under the migration item.
