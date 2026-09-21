# ADR 0048: Shared JPEG raster provider

- Status: Accepted
- Date: 2026-09-20
- Class: [arch] [major]
- Item: [RITK-JPEG-001](../../backlog.md#RITK-JPEG-001)

## Decision

JPEG byte parsing, entropy reconstruction and EXIF interpretation move to
`consus-raster`. RITK retains medical sample signedness, DICOM shape checks,
modality rescaling and image geometry. Its file reader preserves encoded-grid
orientation; EXIF display orientation never substitutes for clinical geometry.

The replaced decoder supported sequential and lossless streams but lacked desktop
progressive decoding. The separate Metis implementation duplicated JPEG parsing.
A common upstream provider removes that duplication without a repository cycle:
RITK's viewer already depends on Metis. Consus owns format parsing and neither
consumer is a dependency of its raster package.

PNG adapters select the PNG codec explicitly for reads and writes. Filename
extensions and content guessing cannot route JPEG bytes through the image
dependency's separate decoder, and a PNG writer emits PNG even when given a
different extension. JPEG file and DICOM entry points use the shared provider.

## Verification

The migration preserves exact lossless integer values, signedness and rescaling,
RGB component layout and grayscale file conversion. Strict malformed/truncated
rejection replaces synthetic entropy fill. Progressive, restart, resource-bound
and EXIF tests live with the shared provider. RITK tests continue to establish
medical conversion and file adapter behavior. Both suites must pass before the
old implementation is removed from the delivered tree.

## Public migration

`decode_jpeg_fragment` retains its medical conversion contract. The public
`jpeg::fixtures` test-input builders are removed with the superseded decoder;
benchmarks keep their fixture construction in benchmark support code. External
test consumers must retain their analytical input fixtures or use
`consus_raster::jpeg::encode_gray` to produce grayscale JPEG input. No forwarding
module remains. This removal is a breaking public change; version publication
remains a separate release action.

The compatibility check against `d46fbda` reports exactly the removed fixture
module and its three public builders (`baseline_fixture`,
`baseline_ycbcr_fixture`, and `baseline_grayscale_fixture`). The other 194 checks
pass; this is compatibility evidence, not decoder correctness evidence.
