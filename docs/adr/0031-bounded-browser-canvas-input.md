# ADR 0031: Bounded browser canvas input adapter

Status: Accepted

Date: 2026-09-11

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

## Context

The RITK browser entrypoints already receive bounded file bytes from Métis and
present RITK-owned frames through its canvas seam. Pointer and wheel input was
still absent, so the browser path could not exercise the shared presentation
reducer or the same cancellation behavior as the native Métis host.

The browser provider reports target-local coordinates and browser wheel units;
RITK's action contract requires finite coordinates, one declared wheel unit and
a viewport mapping for each displayed frame. Three orthogonal canvases also
need an explicit axis route. DICOM parsing, metadata, geometry and viewer
state must remain inside RITK.

## Decision

`WebCanvasPresenter::from_canvas_id_with_input` retains Métis's bounded input
listeners. `take_events` translates pointer and wheel records into
`PresentationEvent`, normalizes line and page deltas to CSS-pixel host units,
rejects unsupported or non-finite values, and ignores non-primary touch
pointers because the current RITK dispatcher has one primary gesture state.

The browser viewer drains each canvas queue once per frame. It constructs a
viewport from the current RITK frame dimensions, routes an orthogonal canvas to
its axis, applies the existing `SnapApp` action reducer, invalidates the frame
only when the reducer requests repaint, and cancels the gesture on translation
or reducer failure. A browser pointer cancel becomes the existing typed
`PointerCancelled` viewer action.

## Rejected alternative

Handling pointer state in Métis would duplicate RITK's viewer reducer and make
clinical interaction semantics part of the GUI host. Passing raw DOM events
would couple RITK to a browser runtime and lose the bounded provider contract.

## Verification

The native RITK action suite covers pointer-cancel reduction and its existing
wheel, viewport and cancellation laws. The RITK WASM target is checked against
the working Metis input seam; the existing packaged synthetic DICOM capture
continues to prove byte-to-frame ownership. Physical pointer, cross-engine,
GPU and full-window captures remain open evidence under the migration item.
