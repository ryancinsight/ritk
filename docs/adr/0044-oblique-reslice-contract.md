# ADR 0044: Physical-plane reslice contract

- Status: Accepted
- Date: 2026-09-20
- Delivery: [RITK PR #539](https://github.com/ryancinsight/ritk/pull/539), merge `2d9377975b4fc420ab0f16481c6e2a6c80279cc7`.

## Context

RITK already preserves DICOM origin, spacing and direction through the VTK
spatial-volume contract and provides exact axis-aligned slab statistics. A
RadiAnt-class viewer also needs an oblique plane whose pixels are defined in
patient space. If each shell derives its own voxel mapping, anisotropic and
rotated studies can display plausible but physically incorrect anatomy. The
host boundary must receive the resulting samples without learning DICOM or
VTK representation details.

## Decision

Add `ReslicePlane` to `ritk-snap::render`. A request names the patient-space
origin of output pixel `[0, 0]`, horizontal and vertical pixel steps, an
optional through-plane step, output dimensions, a finite sample count and an
explicit `ResliceInterpolation` (`Nearest` or `Linear`). Construction builds
one validated `AffineTransform`, rejects non-finite or parallel in-plane
steps, checks all eight request corners against the source voxel box, and
retains the source shape and transform for generation-safe reuse.

`compute_into` maps each requested patient-space point to continuous voxel
coordinates. Nearest sampling selects the rounded voxel. Linear sampling uses
the eight neighbouring scalar values and performs trilinear interpolation in
the source `f32` presentation domain. `ProjectionStatistic` then reduces the
through-plane samples as maximum, minimum or arithmetic average. The output
is a row-major `Vec<f32>` supplied by the caller, bounded to a 64 MiB output
frame and 256 million source evaluations. Repeated renders therefore reuse
storage and cannot allocate from an unchecked request.

`axis_aligned` and `from_slab` derive their patient-space steps from the same
affine and are value-equivalent to `LoadedVolume::extract_slice` for a
one-sample request. The API returns `ResliceOutput`, not a GUI image, DICOM
metadata, or a VTK object. Métis, eframe and future GPU consumers remain
presentation adapters; RITK retains physical geometry and clinical sampling.

This increment does not wire oblique gestures into a host, dispatch slabs to
WebGPU, or move DICOM parsing. Those are separate contracts with their own
input and adapter evidence.

## Alternatives

1. Derive voxel coordinates in each host. Rejected because the affine,
   anisotropic spacing and boundary rules would drift between native, browser
   and future GPU presentations.
2. Add an oblique mode to `SlabProjection`. Rejected because the existing
   request is intentionally integer-axis and exact; mixing continuous plane
   coordinates with its index contract would hide interpolation and bounds
   failures.
3. Put resampling in Metis or `ritk-vtk`. Rejected because sampling a clinical
   volume is RITK domain behavior. VTK remains the geometry carrier and Metis
   remains the format-neutral host.

## Invariants and failure modes

- Only validated scalar volumes are accepted; malformed shape, channel or
  payload declarations return typed errors.
- The source affine is finite and invertible, and a plane's in-plane basis is
  non-zero and non-parallel.
- Every request corner and every sampled coordinate is inside the source
  voxel box, allowing only the affine round-off tolerance documented in the
  implementation.
- Output dimensions and source-evaluation count are checked before sampling;
  caller-owned capacity is retained between computations.
- A changed source shape or physical transform is rejected rather than
  presenting stale pixels.
- DICOM identifiers, source paths, parser objects and host-specific types do
  not cross the reslice contract.

## Verification

Manufactured volumes prove all three axis-aligned planes match the existing
slice extractor, rotated anisotropic coordinates preserve voxel identity, and
trilinear interpolation reproduces a linear field. Slab maximum, minimum and
average are exercised through `from_slab`; invalid basis, out-of-volume,
channel, geometry and payload partitions return typed errors; scratch capacity
is stable after warmup. The locked native and WASM package gates, strict
Clippy, rustdoc, formatting and the existing real-study replay remain the
delivery checks. The committed MRI captures are visual evidence for the
existing orthogonal/MIP hosts; this contract adds no fabricated image.
