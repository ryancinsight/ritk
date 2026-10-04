# ADR 0054: Shared typed stored-volume contract

- Status: Accepted

- Revision 2026-10-04: nonzero NRRD DWI gradients require an explicit
  measurement frame; an all-zero baseline remains valid without one. The
  stored writer emits NRRD0005 with an identity frame for LPS gradients.
  Gzip byte skips count toward expanded-byte limits before decoding. These
  choices follow the [Teem NRRD specification, sections 1.1 and 4](https://teem.sourceforge.net/nrrd/format.html).

This decision is retroactive. It records the shared RITK image-I/O contract
implemented in [PR #749](https://github.com/ryancinsight/ritk/pull/749),
building on the `SampleBuffer` contract in [PR #756](https://github.com/ryancinsight/ritk/pull/756).

## Context

RITK has multiple readers that currently decode stored values directly into
`f32` images. That representation can round wide integers and discard float
payload bits. It also separates values from the physical geometry, coordinate
mapping, and intensity transform needed to write an equivalent target format.
The codec crate knows byte-order and fixed-width representation but does not
own image shape or medical-image calibration.

## Decision

`ritk-image-io` owns `StoredVolume`, a validated 3-D `[depth, row, column]`
value containing a `SampleBuffer`, `ImageMetadata<3>`, `CoordinateMap`, and an
explicit `IntensityCalibration`. Its physical metadata is patient LPS in
millimeters. Construction checks non-empty dimensions, checked sample-count
multiplication, exact buffer length, finite origin and spacing, an invertible
direction matrix, finite nonzero direction-times-spacing components,
coordinate-map rank, and per-frame calibration depth. Small positive spacings
retain their direction; only zero or non-finite lengths use an axis fallback.

Format crates own header parsing and serialization. They exchange
`StoredVolume` for lossless stored-sample reads, writes, and conversions. A
format that cannot preserve a volume's calibration or another required
semantic returns a typed capability error before creating output. A caller
that wants compute-ready values uses the existing image API or an explicit
calibration operation; stored reads do not silently rescale samples.

NRRD maps all ten fixed-width codec sample types and the standard type aliases
to its declared element type, reads both binary payload byte orders, writes
little-endian payloads, preserves exact binary float bits, and separates both leading interleaved and trailing
contiguous acquisition axes. ASCII payloads parse into the declared type; they
do not carry binary float payload bits. The stored NRRD writer emits a trailing
acquisition axis so each volume remains contiguous. Its coordinate-map
extension preserves per-slice transforms as well as fixed-parameter
acquisition maps.
Since NRRD has no standard modality-calibration field, its stored writer
rejects non-identity calibration before opening the output file. The NRRD
type names and payload rules follow the [Teem NRRD format specification,
type and data sections](https://teem.sourceforge.net/nrrd/format.html). Its RAS,
LAS, and LPS basis and physical-unit fields are normalized at the adapter
boundary; supported `space units` and per-axis `units` are converted to
millimeters. Per-axis units without directions or spacings are rejected because
units alone do not define a sample grid. Unknown spaces, anonymous `space
dimension` frames, and units without a known millimeter conversion are also
rejected. The standard `axismins`, `axismaxs`, and `centerings` aliases are
canonicalized before conflict checks. When the optional space or unit field is
absent, the reader retains the existing LPS/mm interpretation. NRRD requires an
explicit `encoding` field. RITK supports `raw`, `ascii` (`text`, `txt`), and
`gzip` (`gz`). ASCII tokens are whitespace-delimited and limited to 128 bytes
per sample to bound parser scratch space. Binary payloads with multi-byte
samples require an explicit `endian` field; one-byte samples and ASCII payloads
do not, following Section 5 of the Teem NRRD specification. Line skips are
applied before payload decoding. Nonnegative byte skips apply after line skips,
and for gzip they apply to decompressed bytes; `byte skip: -1` is supported only
for raw payloads and starts at the file end minus the declared payload length.
Detached data accepts one relative filename and rejects absolute paths and
parent traversal. Stored reads enforce caller-selected encoded-byte,
decoded-byte, and series-volume ceilings before allocating decoded samples;
gzip-expanded bytes include discarded byte-skip data. Defaults are 1 GiB per
byte count and 65,536 volumes. Stored NRRD writes stream exact samples through a
buffered writer instead of allocating a second volume-sized payload. Diffusion
series writes use NRRD0005 and an explicit identity measurement frame; other
series retain NRRD0004. A measurement frame without DWMRI acquisition metadata,
axis support bounds, and cell/node centering are rejected before payload reading
because the stored-volume model cannot retain those semantics.

## Rejected alternatives

- Put stored samples in `ritk-image`: this couples the domain image to file
  representation and format calibration.
- Keep a separate stored-volume type in each adapter: shape, geometry, and
  calibration validation would diverge across formats.
- Decode every format directly to `f32`: wide integers and IEEE 754 payload
  bits cannot be preserved.
- Ignore calibration when the target format cannot represent it: the output
  would decode to a different physical intensity.

## Consequences

The shared crate depends inward on codecs, image metadata, and spatial mapping;
format adapters depend on the shared contract. It does not parse a format or
convert values. Capability declarations and pairwise round-trip tests remain
in each format adapter and the `ritk-io` conversion surface.

## Evidence and revision criteria

`ritk-image-io` tests validate shape and calibration invariants. NRRD tests
read all sample types in both byte orders and round-trip each sample type in
the writer's little-endian encoding, preserve wide integer and floating-point bit patterns, verify both acquisition layouts, and reject
unsupported calibration without changing an existing output. A public-reader
test checks the typed decoded-byte budget error, including gzip byte skips.
Header tests cover standard alias canonicalization, geometry tests reject
per-axis units without a spacing source, and diffusion writer tests assert the
NRRD0005 measurement frame. Revise this decision if a
format's required image semantics cannot be represented by this value without
loss or if an independent format-conversion oracle contradicts these tests.

Its conversion consumer is
[RITK-FORMAT-CONVERSION-001](../../backlog.md#RITK-FORMAT-CONVERSION-001).
