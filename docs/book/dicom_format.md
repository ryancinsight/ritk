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
