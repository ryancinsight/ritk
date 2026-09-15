# ADR 0033: RITK browser trace semantic validator

Status: Accepted

Date: 2026-09-11

Driver: [RITK-SNAP-METIS-002](../../backlog.md#RITK-SNAP-METIS-002).

## Context

Métis emits a schema-1 canvas trace with generic snapshots, actions, PNG
records, and cleanup evidence. It cannot interpret the meaning of a consumer's
attributes. RITK currently publishes a bounded `data-ritk-*` contract, but no
RITK-owned executable validates that a trace contains the required three-view
semantic state and lifecycle evidence. A passing WebDriver process alone is
therefore insufficient evidence for the RITK viewer workflow.

## Decision

Add a native-only `ritk-snap --validate-browser-trace <JSON>` command. The
validator parses the bounded schema needed by RITK, requires a passed canvas
trace with a valid engine and two repository revisions, and checks three
ordered canvases (axial, coronal, sagittal by default). It validates the
RITK-owned attributes, presented dimensions, one trusted pointer and wheel
action per canvas, element and full-window screenshot records, and released
input/cleanup evidence. Custom canvas identifiers may be supplied in the
same order with repeated `--canvas-id` options.

The validator is an evidence consumer, not a browser driver or DICOM parser. It
never reads DICOM bytes, patient data, raw pixel values, or clinical claims.
Pixel and decoded-value assertions remain in RITK's existing workflow tests and
manual captures. Métis remains generic and continues to record only the values
requested by the consumer.

Revision 2026-09-13: the validator now compares the initial and after-input
`data-ritk-slice-index` values for each canvas. A trusted wheel must change the
index when the axis contains more than one slice; a singleton axis is valid at
index zero. The slice count must remain stable across the interaction. This
keeps a trace from passing when the browser reports an input event but the
viewer state and image remain unchanged.

Revision 2026-09-14: `--require-keyboard` selects an explicit keyboard mode for
the same validator. In this mode each declared canvas must report focus plus a
trusted `ArrowDown` keydown/keyup pair targeted at that canvas, with matching
key/code values, `repeat: false`, and no modifier flags. Pointer/wheel-only
traces remain valid under the default mode. The validator still assigns no
shortcut or DICOM meaning to these records; RITK's reducer owns that contract.

Revision 2026-09-14: keyboard-mode traces now carry an `after-keyboard`
semantic snapshot between the focused key pair and the pointer/wheel actions.
The validator requires the slice count to remain stable from the initial state
through that boundary and compares the wheel result with the immediately
preceding snapshot. This prevents a valid keyboard transition followed by an
opposite wheel transition from looking unchanged when only the initial and
final states are compared.

Revision 2026-09-15: hosted run
[34922946179](https://github.com/ryancinsight/ritk/actions/runs/34922946179)
passed `--require-keyboard` validation on Chromium 152 and Firefox 155 for all
three canvases. Safari 26.6.2 failed its bounded file read before a canvas trace
or keyboard action could be produced; the failure is retained as the WebKit
provider regression case.

Revision 2026-09-15 (cine rate): `--require-cine-rate` selects a focused
`=` key/`Equal` code keydown/keyup profile and requires each canvas's
`data-ritk-cine-fps` value to increase after the keyboard action while staying
within the bounded 1–60 FPS contract. The existing `--require-keyboard` mode
continues to validate the ArrowDown navigation profile.

## Rejected alternative

Putting the semantic checks in `scripts/browser_runtime.py` would make the
shared runner own RITK meaning and would duplicate the validator for every
consumer. Relying on screenshots or a successful process exit would not prove
axis order, viewer state, or teardown. Adding a JavaScript callback would widen
the browser trust boundary without improving the evidence contract.

## Verification

Unit tests construct small schema-1 trace values and cover valid three-view
traces plus invalid status, revision, attributes, axis order, dimensions,
actions, keyboard focus/metadata, slice progression, screenshot, cleanup and
oversized-file cases. The committed fixture is validated by the same code used
by the executable. At the current revision,
`cargo nextest run --locked -p ritk-snap` passes 817/817; strict native and
WASM Clippy/check gates and `cargo fmt --all -- --check` pass; and
`cargo run --locked -p ritk-snap -- --validate-browser-trace
crates/ritk-snap/tests/fixtures/browser-trace.json` reports the Chromium
trace and both revisions. Hosted run 34922946179 supplies the same validator
evidence on Chromium and Firefox and records the Safari read failure. The
manual records the command and its limits; configured browser-driver and
clinical pixel evidence remain separate RITK workflow requirements.
