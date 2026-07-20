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

## Invariant

Every DICOM loader must reject before constructing `Image<B,3>` when the
object declares `SamplesPerPixel ≠ 1`.
