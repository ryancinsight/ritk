# 0060 — Locate measurements on the image where they were drawn

Status: Proposed

## Context

An image-space measurement has meaning only for the volume, plane and slice
where the reader placed its points. A saved shape can survive navigation even
when its source slice is no longer visible.

## Decision

Each new annotation stores the RITK plane and zero-based slice index beside
its geometry. A pane draws marks anchored to its image, uses its current
image-to-display transform, and clips marks to that pane. Image readout
visibility does not control measurement visibility.

Session documents written before slice anchoring still deserialize.
Existing lengths with only a plane keep their plane-based presentation.
Earlier angle, region and probe records carry no placement and stay in the
session without appearing on an arbitrary image.

## Alternatives

A separate index-to-location array can drift when an annotation is removed
or reordered. Storing the location with the shape keeps those operations
atomic and makes the serialized entry self-contained.

## Consequences

Rust code that constructs or destructures a measurement variant adds the
`location` field. New measurements use the source pane and slice. Code
recovering an earlier record without that information uses `None`.
This source migration leaves the registry version unchanged.

The native grid test verifies a mark on the recorded slice and its absence
at the same pixel on the adjacent slice. Session tests verify loading earlier
documents without inventing a location.
