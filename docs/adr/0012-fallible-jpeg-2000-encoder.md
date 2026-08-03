# ADR 0012 — Native JPEG 2000 encoding contract

- Status: Accepted
- Date: 2026-07-29
- Board items:
  [SAFE-680-01](../../backlog.md#safe-680-01-major---make-jpeg-2000-encoding-fallible-and-document-the-native-codec),
  [FEAT-692-01](../../backlog.md#feat-692-01-major---control-native-jpeg-2000-lossy-quantization)

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

The irreversible path later added 9/7 lifting and scalar dead-zone
quantization, but its public transform and decomposition arguments were
independent and its quantization step was hardcoded to one. That surface could
express a transform but not its required quantization contract. It also built
a full `f32` wavelet volume and a second full `i32` quantized volume before
EBCOT partitioned the coefficients into code blocks.

## Decision

`encode_grayscale_j2k` returns
`Result<Vec<u8>, Jpeg2000EncodeError>`. It validates all external image
metadata and samples before transform allocation:

- nonzero, checked geometry and exact sample count;
- precision in the implemented 1–16-bit range;
- decomposition depth bounded by the image's resolution geometry; and
- signed or unsigned sample range implied by precision.

Its final argument is one `Jpeg2000Encoding` value:

- `Lossless { decomposition_levels }` selects reversible 5/3 lifting and no
  quantization;
- `Lossy { decomposition_levels, quantization_step }` selects irreversible
  9/7 lifting and requires a positive finite `QuantizationStep`.

The requested lossy step is rounded independently for each subband to the
nearest QCD exponent/mantissa pair permitted by ISO 15444-1 equation E-3. The
exact step reconstructed from that pair is the single source for coefficient
quantization and codestream metadata. A step whose QCD exponent is not
representable for every transformed subband is rejected before transform
allocation. The packet writer retains the transformed `f32` volume and
quantizes one code block at a time, bounding the additional integer workspace
to the nominal 64 × 64 code-block area instead of one image volume.

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
- Irreversible callers can control scalar distortion directly, but the API
  does not claim target size, bitrate, PSNR, or rate-distortion optimization.
- The existing one-component, one-tile, and one-layer limitations remain.

## Rejected alternatives

Keeping the infallible signature and documenting panics leaves untrusted image
metadata on a process-terminating path. Clamping dimensions, precision, or
samples would silently change medical pixel data. Retaining the shifted vector
as a compatibility implementation would preserve avoidable peak memory.

Adding `encode_grayscale_j2k_with_options` beside the old signature would
retain two public contracts and permit the old transform/quantization mismatch.
A percentage quality parameter was rejected because no stable percentage maps
to bitrate or perceptual quality across image content and precision. Target
byte rate control requires multiple truncation points and a rate-distortion
optimizer; labeling scalar quantization as rate control would overstate the
implementation.

## Verification

Value-semantic tests cover each error partition and signed/unsigned sample
boundaries. Reversible round trips and captured OpenJPEG corpus tests remain
exact. The unchanged 512 × 512, five-level Criterion workload provides the
latency comparison. The mdBook example regenerates source, reconstruction, and
absolute-error panels from public API results.

FEAT-692-01 additionally verifies QCD round-trip representation, the analytical
dead-zone error bound, monotonic size reduction on a deterministic medical
phantom, captured OpenJPEG interoperability, and deterministic figure metrics.

## Revision history

- 2026-07-29: Initial accepted decision for SAFE-680-01.
- 2026-08-03: Revised for FEAT-692-01 to make transform and quantization one
  encoding-mode contract, add validated scalar quality control, and bound lossy
  quantization workspace to one code block.
