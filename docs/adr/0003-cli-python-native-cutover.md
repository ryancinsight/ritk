# ADR 0003: CLI and Python native I/O cutover

- Status: Accepted
- Change class: [arch]
- Date: 2026-07-03
- Revision: 2026-09-30 — Phase A now uses shared RITK dispatch for the supported image formats, including MINC2 and derived DICOM output.
- Related: ADR 0002 (core Image Burn-to-Coeus migration), docs/coeus_migration.md

## Context

The processing crates and the public Image API migrate from Burn to the
Atlas-native substrate in dependency order. CLI and Python consumers must use
the native format readers and writers without forcing every processing
command to change at once. Format-specific parsing belongs to its RITK format
crate; conversion and path dispatch need one shared owner.

RITK supports both file formats and domain-specific formats such as DICOM
series. Their metadata and representable sample ranges differ. Conversion
must report those limits and never imply that every pair is lossless.

## Decision

1. ritk-io owns ImageFormat inference, native read/write capability, and the
   shared image dispatch. Format crates own codecs and format semantics.
   Consumers do not maintain parallel format-to-reader or format-to-writer
   matches. write_image_native_with_format handles explicit output selection
   and directory outputs such as DICOM series.

2. Phase A uses this shared path for ritk convert and Python image I/O. DICOM
   input scans and selects a series in RITK; an ambiguous directory requires
   an explicit SeriesInstanceUID. DICOM output is a derived Secondary Capture
   series and does not copy patient or study metadata. Format implementation
   and conversion remain in RITK; Métis receives decoded image data.

3. ImageFormat is non-exhaustive because the RITK format set grows. The
   scalar `f32` conversion dispatch routes NIfTI, MetaImage, NRRD, MINC2, PNG,
   DICOM, MGH, TIFF, VTK, JPEG, and Analyze. PNG and JPEG accept one 2-D slice;
   PNG requires finite integral unsigned 16-bit samples and JPEG is lossy
   8-bit grayscale. DICOM conversion accepts scalar image series and writes a
   derived Secondary Capture series; RGB DICOM uses the separate RITK
   color-volume API. Rank-4 NIfTI, NRRD, and MGH acquisitions use the separate
   series APIs. TIFF omits physical geometry; Analyze does not store direction;
   VTK output requires identity direction. The `f32` carrier cannot represent
   every wide integer sample exactly.

4. The remaining processing commands migrate in dependency order. Statistics
   moves when native statistics are complete; normalize and resample move when
   their native operations cover each command; filter, registration, and
   segmentation move with their own native processing slices. Until then,
   their existing processing-image boundaries remain unchanged. The viewer
   keeps DICOM decoding in ritk-io and presentation in ritk-snap.

## Alternatives considered

- Duplicate format matches in CLI and Python: rejected because support and
  path semantics would drift between consumers.
- Put DICOM parsing or conversion in the GUI shell: rejected because RITK
  owns image formats and presentation hosts must not own a second decoder.
- Convert each file independently inside the processing commands: rejected
  because format dispatch is shared I/O policy, not command-domain behavior.
- Bridge all images back to Burn to preserve the old pipeline: rejected as
  the final migration strategy because it retains a full-image copy at every
  native read. A boundary bridge remains only where an unmigrated processing
  API requires it.

## Consequences

- Adding a 3-D format requires its RITK codec, dispatch route, capability
  entry, consumer selection where exposed, tests, and documentation in one
  change.
- Lossless round-trip claims are format- and sample-specific. Tests assert
  voxels and geometry separately against each format's representable
  semantics.
- Adding ImageFormat variants is a public compatibility change for existing
  exhaustive matches. The migration guide is docs/migration_image_format_dispatch.md.

## Verification

The conversion matrix checks every advertised lossless volume writer and
reader against voxel values and representable geometry. PNG, JPEG, and DICOM
have separate value-semantic tests for their sample, dimensional, and
metadata limits. The RITK I/O capability matrix must agree with actual
dispatch, and CLI help and the repository manual must list the same formats.
