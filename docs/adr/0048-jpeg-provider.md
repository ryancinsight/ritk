# ADR 0048: Shared JPEG raster provider

- Status: Accepted
- Date: 2026-09-20
- Class: [arch] [major]
- Delivery: [RITK PR #556](https://github.com/ryancinsight/ritk/pull/556) and [PR #561](https://github.com/ryancinsight/ritk/pull/561). Merges: `fc85dad03a6c14a617e9687609044497c1eba122`, `71b247c0f0e948614a7e2c5205b34106762c0861`.

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

Revision 2026-09-21: standalone JPEG readers consume Consus's borrowed
`display_samples()` iterator for encoded-precision to eight-bit mapping.
RITK retains grayscale luminance conversion and volume construction; medical
fragment decoding continues to consume raw samples without display mapping.
The downstream Helios DICOM reader supplies required BitsStored through the
same named DICOM tag and `PixelLayout` contract.

## Verification

The migration preserves exact lossless integer values, signedness and rescaling,
12-bit DCT values, RGB component layout and grayscale file conversion. Clinical
decode requires the JPEG frame precision to equal DICOM BitsStored. Signed
lossless samples use that precision's sign bit; signed lossy DCT input is
rejected because its signed interpretation is ambiguous. Compressed sample
width remains independent of the BitsAllocated container width, so 2–8-bit
lossless samples remain exact in 16-bit DICOM containers. DICOM readers reject
missing or malformed required BitsStored metadata rather than inferring a
precision. Strict
malformed/truncated rejection replaces synthetic entropy fill. Progressive,
restart, resource-bound and EXIF tests live with the shared provider. RITK tests
continue to establish medical conversion and file adapter behavior. Both suites
must pass before the old implementation is removed from the delivered tree.

## Public migration

`decode_jpeg_fragment` retains its medical conversion contract. The public
`jpeg::fixtures` test-input builders are removed with the superseded decoder;
benchmarks keep their fixture construction in benchmark support code. External
test consumers must retain their analytical input fixtures or use
`consus_raster::jpeg::encode_gray` to produce grayscale JPEG input. No forwarding
module remains. This removal is a breaking public change; version publication
remains a separate release action.

`PixelLayout` gains the required `bits_stored` field so allocated container
width is distinct from JPEG sample precision. The public DICOM slice and
multiframe metadata types carry the same value through every frame request.
External struct literals must provide it. This is an additional breaking API
change in the same major migration; manifest version changes remain reserved
for the separately authorized release action.

The compatibility check against `d46fbda` reports the required `PixelLayout`
field and removed fixture module/builders as three major-change classes.
The other 193 checks pass and 58 do not apply. This is compatibility evidence,
not decoder correctness evidence.
