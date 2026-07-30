# ADR 0012 — Fallible native JPEG 2000 encoder

- Status: Accepted
- Date: 2026-07-29
- Board item: [SAFE-680-01](../../backlog.md#safe-680-01-major---make-jpeg-2000-encoding-fallible-and-document-the-native-codec)

## Context

The public grayscale JPEG 2000 encoder accepted external geometry, precision,
decomposition depth, and samples but returned `Vec<u8>`. It asserted sample
length and precision, multiplied dimensions without checked arithmetic, and
accepted samples outside the declared component range. Unsigned encoding also
allocated a complete DC-shifted image before the packet writer allocated its
transform buffer.

The native entropy and reversible lifting representation is currently `i32`.
DICOM PS3.5 permits Bits Stored values up to 38 for JPEG 2000, but claiming that
range through this API would require a wider coefficient representation.

## Decision

`encode_grayscale_j2k` returns
`Result<Vec<u8>, Jpeg2000EncodeError>`. It validates all external image
metadata and samples before transform allocation:

- nonzero, checked geometry and exact sample count;
- precision in the implemented 1–16-bit range;
- decomposition depth bounded by the image's resolution geometry; and
- signed or unsigned sample range implied by precision.

The packet writer receives the validated source slice and a DC offset. It
applies the offset while constructing the transform buffer, eliminating the
separate full-image shifted allocation. Every in-repository caller adopts the
fallible API in the same change; no infallible compatibility wrapper remains.

## Consequences

- Invalid caller input returns a contextual, matchable error and cannot produce
  a partial codestream.
- Reversible encoding performs one fewer full-image allocation and write pass.
- The public API change requires a major release when it is published.
- Precision above 16 bits is rejected explicitly until the wavelet,
  quantization, and EBCOT coefficient representation is widened and verified.
- The existing one-component, one-tile, one-layer, no-rate-control limitations
  remain unchanged.

## Rejected alternatives

Keeping the infallible signature and documenting panics leaves untrusted image
metadata on a process-terminating path. Clamping dimensions, precision, or
samples would silently change medical pixel data. Retaining the shifted vector
as a compatibility implementation would preserve avoidable peak memory.

## Verification

Value-semantic tests cover each error partition and signed/unsigned sample
boundaries. Reversible round trips and captured OpenJPEG corpus tests remain
exact. The unchanged 512 × 512, five-level Criterion workload provides the
latency comparison. The mdBook example regenerates source, reconstruction, and
absolute-error panels from public API results.

## Revision history

- 2026-07-29: Initial accepted decision for SAFE-680-01.
