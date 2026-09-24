# ADR 0052: Native oblique MPR presentation

- Status: Accepted
- Date: 2026-09-24
- Item: [RITK-SNAP-OBLIQUE-NATIVE-001](../../backlog.md#RITK-SNAP-OBLIQUE-NATIVE-001)

## Context

ADR 0044 defines a host-neutral physical reslice plane but stops before host
gestures. The Métis native viewer currently maps pointer input only through
axis-aligned viewports. Applying that mapping to an oblique frame produces
plausible but incorrect linked slices and measurements.

## Decision

RITK owns oblique plane construction, scalar sampling, output-pixel mapping,
and patient-space measurements. The native shell places the resulting RGBA
frame in the fourth pane and maps pointer positions through the exact
`ReslicePlane` used to render it. Wheel input translates the plane along its
normal; arrow keys change its orientation; clicks update the linked voxel
cursor. Native presentation adds a separate launch entrypoint and CLI layout,
keeping the public exhaustive `NativePresentationMode` unchanged.

The plane is centred on a validated voxel, uses an orthogonal in-plane basis
derived from the volume affine, and chooses a bounded rectangular extent whose
corners remain inside the volume. Pixel size is derived from the physical
voxel spacing. Resize, zoom, pan, and rotation affect placement or the plane
basis without changing the source geometry contract.

Completed oblique lengths store validated patient-space endpoints and the
derived millimetre distance. `ViewerSessionSnapshot` writes format 3 while
the reader continues to accept formats 1 and 2. `Annotation` becomes
non-exhaustive; consumers migrate by handling the patient-space length variant
or adding a wildcard arm. This is a major API change, delivered without a
registry release.

## Alternatives

1. Reuse `ViewerViewport` with a fabricated axis. Rejected because its axis
   determines voxel and slice semantics.
2. Convert oblique pixel distance through two scalar spacings. Rejected
   because a rotated plane has a patient-space basis that scalar axis spacing
   cannot represent.
3. Add a variant to `NativePresentationMode`. Rejected to preserve the
   existing exhaustive public enum; the new native launch entrypoint carries
   the distinct workflow.

## Consequences

- RITK remains the only owner of DICOM geometry and clinical sampling; Métis
  receives pixels, host coordinates, and presentation controls.
- Patient-space annotations survive session serialization and remain
  interpretable after the plane rotates.
- The eframe and browser shells retain their existing orthogonal workflows;
  browser oblique presentation is a dependent item using the same RITK
  coordinate contract.
- A failed plane rebuild or sample is surfaced and does not replace the last
  valid frame.

## Migration

The `Annotation` enum gains a patient-space length variant and becomes
`#[non_exhaustive]`. Downstream exhaustive matches add a wildcard or handle
the new variant. Session writers emit format 3; readers continue to accept
formats 1 and 2 and validate format-3 endpoints. No registry release is part
of this change.

## Verification

A rotated, anisotropic, non-square manufactured volume supplies exact
pixel-to-patient-to-voxel and linked-cursor oracles. A 3–4–5 patient-space
segment verifies length. Native pointer tests cover pan, zoom, wheel, key
orientation and invalid boundaries. The public MRI-DIR phantom capture is
the visual check for the complete host workflow.
