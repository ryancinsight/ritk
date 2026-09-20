# ADR 0047: Browser annotation result semantics at the Métis boundary

- Status: Accepted
- Item: [RITK-SNAP-METIS-ANNOTATIONS-001](../../backlog.md#RITK-SNAP-METIS-ANNOTATIONS-001)

## Context

RITK already computes Length, Angle, ROI and HU annotations from the decoded
study. The browser gallery could prove that a tool changed the rendered frame
and that its button was selected, but it could not prove that the gesture
completed the input-sensitive calculation. Reading annotation state through a
consumer-only JavaScript side channel would duplicate viewer state and would
not cover native presentation snapshots.

## Decision

`PresentationSnapshot` carries the completed annotation count and an optional
summary of the latest annotation. The summary contains a stable kind label and
one primary computed value: millimetres for Length, degrees for Angle, square
millimetres for both ROI kinds, and the sampled intensity for HU Point. The
constructor asserts that the published value is finite. RITK remains the owner
of annotation math, DICOM pixels and units.

Browser canvases publish the summary as three bounded attributes:
`data-ritk-annotation-count`, `data-ritk-last-annotation-kind` and
`data-ritk-last-annotation-value`. The gallery tool trace reads those values
from all three canvases, requires agreement, and checks the expected count and
kind after each real measurement gesture. Non-measurement tools must preserve
the summary. Empty state uses count `0` and empty kind/value attributes.

The browser replay performs all five measurement gestures before pan, zoom and
window-level gestures. Those viewport tools change the displayed coordinate
mapping; running them first would make fixed-coordinate measurement probes land
on the padded viewport rather than the decoded anatomy. RITK PR [#552](https://github.com/ryancinsight/ritk/pull/552)
fixes that replay ordering and keeps the input-sensitive assertions unchanged.

## Alternatives

* Recompute measurements in JavaScript. Rejected because it duplicates spacing,
  sampling and unit semantics outside RITK.
* Publish the complete annotation list. Rejected because the browser contract
  needs only bounded evidence for the latest transition and would otherwise
  expose unbounded clinical state.
* Treat a changed frame as proof of a result. Rejected because rendering can
  change without a completed annotation and does not establish the computed
  value.

## Evidence and limits

Rust snapshot and browser-semantic tests cover empty, finite and latest-value
projection; the browser tool tests reject non-finite values, require
input-sensitive transitions and replay measurements before viewport changes.
The hosted Chromium tool replay is the visual evidence over the real 94-file
MRI study and records the result transitions and source digests. WebKit bounded
file reads and Chromium WebGPU adapter availability remain separate
host-capability residuals.
