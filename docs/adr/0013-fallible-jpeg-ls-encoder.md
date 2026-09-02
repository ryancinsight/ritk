# ADR 0013: Fallible native JPEG-LS encoder and rolling reconstruction rows

- Status: Accepted
- Date: 2026-07-30
- Board item: [SAFE-681-01](../../backlog.md#safe-681-01-major---make-jpeg-ls-encoding-fallible-bound-header-parsing-reduce-scan-memory-and-document-the-native-codec)

## Context

The public grayscale JPEG-LS encoder accepted external dimensions, precision,
samples, and near-lossless error bound but returned `Vec<u8>`. It asserted
sample count, precision, and range, multiplied dimensions without checked
arithmetic, and narrowed dimensions and `NEAR` into SOF55/SOS fields. Invalid
caller input could therefore panic or produce a header that did not describe
the encoded image.

Both entropy paths retained `(rows + 1) × cols` reconstructed `i32` samples.
The JPEG-LS causal neighborhood defined by ITU-T T.87 Annex A reads only the
previous row and the reconstructed prefix of the current row. DICOM PS3.5
section 8.2.3 constrains the supported monochrome precision to 2–16 stored
bits and identifies the lossless and near-lossless transfer syntaxes.

The SOS parser also converted an unknown interleave byte to the non-interleaved
mode, skipped truncated marker segments, and then rescanned the stream to find
scan data. Those paths could admit malformed external input as a different
stream profile or lose the marker that actually violated the contract.

## Decision

`encode_grayscale_jpeg_ls` returns
`Result<Vec<u8>, JpegLsEncodeError>`. Validation precedes entropy allocation:

- dimensions must be nonzero and fit the 16-bit SOF55 fields;
- the checked pixel count must equal the sample slice length;
- encoder precision remains the implemented 8–16-bit grayscale range;
- `NEAR` must fit the SOS byte and be no greater than `MAXVAL / 2`; and
- every sample must fit the declared precision.

Encoder and decoder use one shared `ReconstructionRows` component backed by a
single `2 × cols` allocation. It owns boundary guards and row rotation so both
entropy directions use the same causal-neighborhood convention. The decoder
validates 2–16-bit precision and the precision-dependent `NEAR` bound before
constructing coding parameters. One bounds-checked marker pass returns the
exact scan slice. Invalid segment lengths, SOS interleave values, mapping-table
selectors, restart intervals, and unsupported LSE parameter IDs return errors.

Every in-repository encoder caller adopts the fallible API in the same change.
No infallible wrapper or parser fallback remains.

## Consequences

- Invalid metadata returns a contextual, matchable error without panic,
  truncation, or partial codestream output.
- Reconstruction scratch decreases from
  `(rows + 1) × cols × sizeof(i32)` to `2 × cols × sizeof(i32)`. A 512 × 512
  frame decreases from 1,050,624 bytes to 4,096 bytes.
- Scratch is independent of image height and uses one allocation per scan.
- The public return-type change requires a major release when published.
- Color and interleaved JPEG-LS remain unsupported and are not hidden behind
  fallback behavior.

## Rejected alternatives

Keeping the infallible signature and documenting assertions leaves external
medical-image metadata on a process-terminating path. Clamping values would
silently alter image geometry or the required error bound. Retaining the full
reconstructed plane provides no information used by the causal predictor.
Mapping an invalid interleave byte to zero changes malformed input into a
different declared profile.

## Verification

Value-semantic tests cover each validation partition, bounded marker parsing,
SOS interleave and unsupported-profile rejection, two-row workspace size and
rotation, exact lossless reconstruction, the exact
`|decoded - original| <= NEAR` bound, randomized round trips, and DICOM
encapsulation. A bounded arbitrary-byte parser property rejects panics. The
unchanged 512 × 512 Criterion workloads compare latency. The mdBook example
regenerates source, lossless, near-lossless, and magnified error panels from
public API output.

## References

- ITU-T T.87, *Lossless and near-lossless compression of continuous-tone
  still images*, Annex A:
  <https://www.itu.int/rec/T-REC-T.87-199806-I/en>
- DICOM PS3.5, section 8.2.3, *JPEG-LS Image Compression*:
  <https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_8.2.3.html>

## Revision history

- 2026-07-30: Initial accepted decision for SAFE-681-01.
