# 0053: Typed stored-sample I/O

Status: Accepted

This decision is retroactive. It records the stored-sample boundary required
by `RITK-TYPED-SAMPLES-001` and the format-conversion campaign
`RITK-FORMAT-CONVERSION-001`.

## Context

RITK owns its medical and scientific image-format readers, writers, and
conversions. The existing codec helpers often decode directly to `f32`, which
can round wide integers and discard floating-point payload bits. Algorithm
scalar requirements belong to image computation; they do not define the
sample representation stored by a file format. DICOM and NIfTI can also carry
intensity calibration, and formats carry physical geometry, which must remain
available to conversion code.

## Decision

`ritk-codecs` owns a sealed `Sample` contract for fixed-width signed and
unsigned integers and IEEE 754 floats. `SampleBuffer` decodes and encodes the
stored representation in either byte order, rejects partial samples, and
requires an exact type match for extraction. A failed extraction retains the
original samples. This layer has no Coeus scalar bound.

Format adapters own geometry, calibration, and other format metadata beside
the stored samples. Numeric conversion, including applying modality scaling,
occurs only through an explicit format or image operation that preserves its
policy and failure mode. Shared image conversion belongs in RITK's image-I/O
layer; GUI consumers select and present its results and do not parse or
convert file formats.

The existing `byte_decode` API remains until every caller is migrated in
dependency order. It is not a second long-term conversion path.

## Rejected alternatives

- Decode every format directly to `f32`: this cannot represent every 32- or
  64-bit integer exactly and cannot preserve all float bit patterns.
- Make codec samples implement Coeus `Scalar`: this couples file storage to
  algorithm support and excludes stored types the compute layer need not use.
- Keep a separate numeric-conversion loop in each format adapter: the same
  type and byte-order rules would drift across readers and writers.
- Apply scaling implicitly during stored-sample decoding: it changes samples
  and loses the distinction between stored values and calibrated intensities.

## Consequences

The typed buffer provides one codec boundary for the format adapters. Image
conversion must carry sample type, physical geometry, and format calibration
until an explicit target-format policy consumes or represents them. A target
that cannot represent required semantics returns a typed error before writing
partial output. The shared capability and conversion contracts are delivered
by `RITK-FORMAT-CONVERSION-001`.

## Evidence and revision criteria

The boundary is exercised by endian round trips, wide-integer values beyond
binary32's exact-integer range, float signed-zero and NaN-payload bit checks,
partial-sample rejection, and mismatch recovery. The decision changes if an
independent format-conversion oracle demonstrates that this stored-sample
boundary cannot preserve a supported format's value or metadata semantics.

Driving item: `RITK-TYPED-SAMPLES-001` (PR #696).
