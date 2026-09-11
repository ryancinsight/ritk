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

## Rejected alternative

Putting the semantic checks in `scripts/browser_runtime.py` would make the
shared runner own RITK meaning and would duplicate the validator for every
consumer. Relying on screenshots or a successful process exit would not prove
axis order, viewer state, or teardown. Adding a JavaScript callback would widen
the browser trust boundary without improving the evidence contract.

## Verification

Unit tests construct small schema-1 trace values and cover valid three-view
traces plus invalid status, revision, attributes, axis order, dimensions,
actions, screenshot, cleanup and oversized-file cases. The committed fixture is
validated by the same code used by the executable. At the accepted revision,
`cargo nextest run --locked -p ritk-snap` passes 812/812, warning-denied
all-target Clippy and
`cargo fmt --all -- --check` pass, and
`cargo run --locked -p ritk-snap -- --validate-browser-trace
crates/ritk-snap/tests/fixtures/browser-trace.json` reports the Chromium
trace and both revisions. The manual records the command and its limits;
configured browser-driver and clinical pixel evidence remain separate RITK
workflow requirements.
