# ADR 0045: Host-neutral viewer presentation snapshot

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-SNAP-PRESENTATION-SNAPSHOT-001](../../backlog.md#RITK-SNAP-PRESENTATION-SNAPSHOT-001)

## Context

RITK owns DICOM decoding, physical geometry, display policy and interaction
state. The native Métis session and the browser canvas both publish evidence
about that state, but separate host-facing structs had started to derive the
same fields independently. That shape can let a browser semantic attribute and
a native observation disagree after a state transition even when both render
the same pixels. The host must also remain format-neutral: no path, DICOM
identifier, metadata object or pixel buffer belongs in the shared state view.

## Decision

`ritk_snap::presentation::PresentationSnapshot` is the single value-semantic
projection of host-visible viewer state. It carries the visual revision,
loaded state, active axis and all three slice selections/counts, effective
window/level, cine state and rate, viewport zoom/pan, optional active window
preset, and the stable interaction-tool index/label. Its fields remain private
and consumers use typed getters. Construction stays in `SnapApp`, where the
viewer state and its invariants are available.

Browser canvas semantics retain one snapshot and add only the presented frame
dimensions. Orthogonal browser canvases derive their per-canvas axis from the
same snapshot before publishing DOM attributes. The native session records the
same snapshot beside its framebuffer observation and exposes it through
`NativeViewerOutcome::snapshot`. Neither host re-derives clinical state.

The snapshot is format-neutral and does not include DICOM paths, identifiers,
metadata, parser values, volume storage or pixels. DICOM semantics remain in
RITK's loader and render layers; Métis receives only the retained presentation
frame and bounded host events.

## Alternatives

1. Keep separate native and browser state structs. Rejected because duplicated
   projections can drift at the host boundary and require parallel tests.
2. Move the projection into Métis. Rejected because the projection reads
   clinical viewer state and would move RITK ownership across the boundary.
3. Expose the full `SnapApp` or `LoadedVolume`. Rejected because it would leak
   DICOM data, storage and mutable domain state into host adapters.

## Invariants and failure modes

- Every snapshot axis is one of the three orthogonal axes and every slice
  count is nonzero.
- Cine rate and zoom are finite and bounded by their existing viewer contracts;
  pan coordinates are finite.
- Browser and native observations are refreshed whenever a host-visible state
  transition is recorded, including the initial session state.
- A missing native snapshot at orderly session completion is an error rather
  than a fabricated empty observation.
- Browser DOM state contains only the existing bounded scalar attributes and
  frame dimensions; no clinical identifiers or pixel payloads cross the seam.

## Verification

The focused `ritk-snap` native suite asserts empty and loaded snapshot
projections, anisotropic slice counts, window/level, cine, zoom, pan, active
tool and revision changes. Browser semantic tests consume the same snapshot;
native-session tests inspect the recorded snapshot after navigation. The
locked native and WASM library checks, strict Clippy, rustdoc, formatting and
the existing real-study replay remain the delivery gates. The committed MRI
manual captures remain the visual ground truth; this contract adds metadata
about the state that produced those real pixels and does not fabricate images.
