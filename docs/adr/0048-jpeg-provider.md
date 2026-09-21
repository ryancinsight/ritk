# ADR 0048: Shared JPEG raster provider

- Status: Accepted
- Date: 2026-09-20
- Item: [RITK-JPEG-001](../../backlog.md#RITK-JPEG-001)

## Decision

JPEG byte parsing, entropy reconstruction and EXIF interpretation move to
`consus-raster`. RITK retains medical sample signedness, DICOM shape checks,
modality rescaling and image geometry. Its file reader preserves encoded-grid
orientation; EXIF display orientation never substitutes for clinical geometry.

The existing decoder supports sequential and lossless streams but lacks desktop
progressive decoding. The separate Metis implementation duplicates JPEG parsing.
A common upstream provider removes that duplication without a repository cycle:
RITK's viewer already depends on Metis. Consus owns format parsing and neither
consumer is a dependency of its raster package.

## Verification

The migration preserves exact lossless integer values, signedness and rescaling,
RGB component layout and grayscale file conversion. Strict malformed/truncated
rejection replaces synthetic entropy fill. Progressive, restart, resource-bound
and EXIF tests live with the shared provider. RITK tests continue to establish
medical conversion and file adapter behavior. Both suites must pass before the
old implementation is removed from the delivered tree.
