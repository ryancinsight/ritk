# DICOM Format Boundary

Single source of truth for DICOM file parsing and pixel-frame decode.

## Ownership

`ritk-io::format::dicom` owns the DICOM Part 10 file parser and pixel
frame decoder. `ritk-dicom` provides the backend trait implementations.

## Boundary Surface

- `DicomParseBackend`: parses a Part 10 file into a backend-owned object.
- `PixelDecodeBackend`: decodes one frame from a backend-owned object using
  `DecodeFrameRequest`.
- `DicomBackend`: combines parse and decode without dynamic dispatch.

## Spatial Contract

DICOM file-axis `[x,y,z]` maps to RITK `[depth,row,col]` via `spatial.rs`.
Physical-space metadata (origin, spacing, direction) is preserved through
the boundary.

## Codec Ownership

`ritk-codecs` owns JPEG, JPEG-LS, JPEG 2000, RLE, PackBits, and native
pixel primitive implementations. Native-owned JPEG syntaxes route exclusively
through `NativeCodecBackend`.

## Pixel Precision

Frame requests keep DICOM BitsAllocated and BitsStored separate. BitsAllocated
selects the byte container; BitsStored identifies the meaningful magnitude and
sign bits. Encapsulated JPEG, JPEG-LS, and JPEG 2000 headers must agree with
BitsStored before samples reach the modality transform. Lossless signed JPEG
uses the codestream precision's sign bit. Lossy DCT JPEG with signed DICOM
metadata is rejected because the transform output does not define a signed
stored-sample representation.

Image readers reject a missing or malformed BitsStored attribute instead of
inferring it from BitsAllocated. When HighBit is present, the DICOM object
boundary requires it to equal BitsStored minus one, which establishes the
right-justified sample layout used by native decoding.

`decode_stored_pixel_frame` returns typed stored integers before modality rescale;
display decoding remains a separate rescaled `f32` path.

## Stored-series import

`read_dicom_stored_series` reads one scanned image series into a
`StoredSeries`. `load_dicom_stored_series` accepts a scanner-produced series
descriptor and requires its retained Part 10 bytes, so pixel decoding uses the
same validated input that supplied the metadata. The result keeps signed or
unsigned stored integer values and the physical position of each slice in a
`SliceSeries` coordinate map; it does not resample the source geometry.

Native implicit-VR and explicit-VR little-endian, single-frame monochrome
instances are supported. Pixel encoding must agree across slices. Linear
modality slope and intercept remain per-slice calibration, while a single-item
Modality LUT remains a typed lookup calibration. When the Modality LUT Module
is present, DICOM permits the LUT or slope/intercept form, not both, and defines
the LUT descriptor, output precision, and value-unit type in the [Modality LUT
Module, PS3.3 C.11.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.html)
and [Little Endian transfer syntax, PS3.5 A.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_A.2.html).
The source `RescaleType` or sequence-item `ModalityLUTType` is retained as an
uninterpreted intensity-unit label on the shared `StoredVolume`; the original
DICOM elements also remain available through per-slice preservation metadata.
Odd-length native pixel values may carry one zero padding byte; truncated data
and nonzero padding fail preflight.

Encapsulated or big-endian transfer syntaxes, color pixels, and multi-frame
instances return typed errors before a volume is built. This API imports one
selected series; combining multiple study series remains a separate operation.

## Pixel output

Series writers encode unsigned 16-bit samples. Multi-frame output uses
unsigned 8-bit samples for baseline JPEG and unsigned 16-bit samples for
native, JPEG-LS, JPEG 2000, lossless JPEG, and RLE transfer syntaxes.
BitsAllocated and BitsStored equal that sample width, HighBit equals the
width minus one, and PixelRepresentation is zero. These attributes describe
the encoded samples as required by [DICOM PS3.5 section 8.1.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_8.html#sect_8.1.1);
compressed fragment lengths describe the codestream, not the decoded sample
width. Source bit attributes are checked for consistency but never copied
over the output representation.

RT Dose emits unsigned 32-bit samples after checking the supplied positive
dose scaling and each rounded sample's range. SEG derives one-bit or eight-bit
storage from BINARY or FRACTIONAL segmentation, rejecting a contradictory
declared width. BINARY pixels follow the least-significant-bit-first stream in
[PS3.5 D.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_D.html):
frames share boundary bytes, and padding follows the complete value. The former
MSB-first, per-frame-padded reader and writer contradicted this wire contract;
both now use the specified packing.

Each series slice has its own linear rescale; a multi-frame object has one
rescale for the entire volume. Constant input encodes as zero with the input
constant as intercept. Nonconstant input maps its finite range onto the full
unsigned sample range with nearest-integer rounding. Non-finite samples,
unrepresentable ranges, invalid dimensions, malformed source bit attributes,
and invalid spatial values return a `DicomWriteError` cause through the
existing writer result. Decimal String components fit the 16-byte wire limit.
Spacing is positive and direction cosines are orthonormal within a bound
derived from one rounding of each component to single precision followed by
the double-precision dot product. No orientation normalization occurs silently.

Writers prepare and serialize all output before creating a directory or
replacing any file. This preflight retains the serialized volume in memory
and preserves existing output on input or serialization failure. Filesystem
failures during persistence can still leave partial output; this is not an
atomic transaction across a series.

## Diffusion metadata

`read_dicom_gradient_scheme_from_file` reads one classic single-frame volume,
while `read_dicom_gradient_scheme_from_files` accepts one representative file
per volume in explicit acquisition order. The reader uses only the standard
top-level Diffusion b-value `(0018,9087)` and Diffusion Gradient Orientation
`(0018,9089)` attributes defined by [DICOM PS3.3
C.8.13.5.9](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.13.5.9.html).
It validates finite s/mm² values and three finite direction components, then
constructs a physically typed LPS `GradientScheme`. It does not infer volume
grouping from a directory or guess private vendor tags; enhanced functional
groups require a separate sequence-aware reader.
For an unweighted frame, a finite zero b-value with no orientation is mapped
to the required zero vector; nonzero weighting still requires an orientation.

## Invariant

Every DICOM loader must reject before constructing `Image<B,3>` when the
object declares `SamplesPerPixel ≠ 1`.
