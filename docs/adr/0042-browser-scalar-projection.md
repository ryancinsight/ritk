# ADR 0042: Browser scalar projection surface

- Status: Accepted
- Date: 2026-09-20
- Driver: [RITK-SNAP-BROWSER-PROJECTION-001](../../backlog.md#RITK-SNAP-BROWSER-PROJECTION-001)

## Context

The browser presentation already exposes three interactive RITK canvases for
the axial, coronal and sagittal planes. The native Métis workflow now also
demonstrates scalar maximum, minimum and average slab statistics. A browser
consumer needs the same real-study projection evidence without moving voxel
reduction or DICOM semantics into Métis.

The existing three-canvas entrypoint is consumed by the checked-in gallery and
hosted traces. Changing its arity or silently adding a fourth listener would
break that contract. A separate four-canvas entrypoint can preserve the
existing API while making the additional panel explicit.

## Decision

Add `start_web_orthogonal_canvases_with_projection` and its explicit WebGPU
counterpart. Both accept one validated statistic index (`0` maximum, `1`
minimum, `2` average) and four canvas identifiers in axial, coronal, sagittal,
projection order. The parser rejects non-finite, fractional and out-of-range
values before any browser surface is mounted.

The first three canvases retain the existing input listeners and semantic
contract. The projection canvas is display-only: it resolves a Métis canvas
without input listeners, publishes a bounded projection role/statistic/frame
attribute set, and receives a RITK `PresentationFrame` only after the typed
`SlabProjection` has reduced the full axis-0 scalar slab. RITK then applies its
existing grayscale window/level, colormap and physical-spacing policy. The
projection frame uses the same reusable RGBA scratch storage as orthogonal
frames; the scalar reduction keeps a separate reusable `Vec<f32>` because the
typed slab contract intentionally accepts a standard vector sink.

The default three-canvas and single-canvas exports remain unchanged. The
WebGPU entrypoint is opt-in and returns setup errors without switching to the
raster provider. Color volumes reject the projection request with the existing
typed scalar contract error.

## Alternatives

1. Add a fourth canvas to `start_web_orthogonal_canvases`. Rejected because it
   changes the established gallery arity and listener/semantic oracles.
2. Add three statistic-specific exports. Rejected because the statistic is a
   real variation dimension already represented by `ProjectionStatistic`; one
   validated index keeps the public surface singular.
3. Let Métis compute or label the projection. Rejected because Métis is
   format-neutral and must not receive DICOM or voxel semantics.

## Invariants and failure modes

- The projection statistic is validated before browser listeners or animation
  tasks are created.
- Only scalar volumes enter the slab reduction; malformed or changed shapes
  fail through `SlabProjection` rather than being clamped.
- The three interactive canvases preserve their existing axis, slice, input,
  window/level and cine attributes.
- The projection canvas has zero input listeners and publishes its statistic,
  load/frame state, dimensions and physical display aspect.
- Maximum output uses the same `SlabProjection` and scalar mapping as the
  native projection contract; no browser-only voxel indexing exists.

## Verification

Parser tests cover valid statistics and finite/integer/range rejection. The
WASM library check and strict Clippy compile both new exports, the projection
surface tests assert listener and frame contracts, and the existing native
slab/native-session tests retain the real public CT pixel evidence. Hosted run
[`35500085568`](https://github.com/ryancinsight/ritk/actions/runs/35500085568)
then exercised the current RITK/Metis pair: Chromium accepted the 94-file
MRI-DIR study and passed the MIP projection artifact
([`10601916572`](https://github.com/ryancinsight/ritk/actions/runs/35500085568/artifacts/10601916572)).
The revision-bound `projection.json` proves 512 × 512 dimensions, 110,028
non-black pixels, MIP attributes, `consumer_listeners: 21`, and
`display_only: true`; the element screenshot's RGBA digest is
`470e898c9dcd60a800155a29cfcd70acd72a597be7c98ac24d4a0d5e963d8924`.
The same matrix records WebGPU no-adapter and WebKit file-read limits without
claiming a raster or sandbox fallback.
