# TIFF and BigTIFF Format Boundary

TIFF is a tagged raster container rather than one fixed pixel encoding. A file
starts with a byte-order and version header, then links one or more Image File
Directories (IFDs). Each IFD describes one raster through tags such as width,
height, sample representation, photometric interpretation, compression, and
strip or tile locations. The [TIFF 6.0 specification](https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.html)
is the format reference.

Classic TIFF stores file offsets in 32 bits. BigTIFF changes the header version
and directory layout to use 64-bit offsets while retaining the same image-data
model. The [BigTIFF design](https://bigtiff.org/) documents the header and IFD
differences. RITK reads both forms through the same API. Its public writer emits
classic TIFF.

## RITK's volume model

TIFF defines a sequence of images, not a medical three-dimensional coordinate
system. RITK gives that sequence one explicit interpretation:

```text
IFD page 1    IFD page 2                     IFD page N
    │             │                               │
    ▼             ▼                               ▼
  z = 0         z = 1             ...         z = N - 1

each page: rows outside, columns inside  →  Image shape [z, y, x]
```

Every page must have the same width, height, and accepted color model. Page
order is the linked IFD order. RITK does not sort pages by a tag or filename.

The public surfaces are:

| Operation | Accepted representation | Returned representation |
|---|---|---|
| `read_tiff` | grayscale TIFF or BigTIFF page stack | `Image<f32, B, 3>` with shape `[z, y, x]` |
| `read_tiff_color_to_volume` | RGB TIFF or BigTIFF page stack | `RgbVolume<f32, B>` with shape `[z, y, x, 3]` |
| `write_tiff` | `Image<f32, B, 3>` | one classic `Gray32Float` IFD per z-slice |
| `TiffReader`, `TiffColorReader`, `TiffWriter` | backend-bound adapters | same contracts as the functions |

Palette, grayscale-alpha, RGBA, CMYK, and YCbCr pages are rejected by these
loaders. The grayscale loader also rejects an RGB page that appears later in
an otherwise grayscale stack; a matching sample count cannot substitute for a
matching color contract.

## Scalar conversion

The grayscale reader accepts decoded unsigned integers, signed integers, and
IEEE `f32` or `f64` pages. Every value becomes `f32` in the returned image.
The conversion is exact for every `u8`, `u16`, `i8`, and `i16` value. Larger
integers can round outside binary32's exact integer range, and finite `f64`
values can round or overflow to infinity. The RGB reader applies the same
conversion independently to its three explicit channels.

The writer stores `Gray32Float`, so an `f32` voxel stack round-trips bit for
bit. It does not quantize to an integer TIFF representation.

## Physical-space metadata

The current RITK TIFF API does not encode or interpret a complete medical
origin, three-axis spacing, and direction matrix. Reading therefore assigns:

```text
origin    = [0, 0, 0]
spacing   = [1, 1, 1]
direction = identity
```

Writing preserves pixels and page order but omits the source image's RITK
geometry. This is an intentional boundary, not a registration result. A caller
must restore geometry from an authoritative companion source before combining
the volume with scanner-space DICOM, NIfTI, MGH, or another physical image.

## Bounded decoding and memory

RITK validates each page's color model and checked `width × height × channels`
sample count before asking the decoder for its raster. Every decoded page is
then appended directly into the final `Vec<f32>` after a fallible reserve:

- an `f32` first page becomes the final allocation directly;
- later `f32` pages append to that allocation;
- integer and `f64` pages convert from the decoder-owned page straight into
  final storage, without constructing a second page-sized `Vec<f32>`.

The locked `tiff` 0.9.1 decoder also applies per-image, intermediate-buffer,
and IFD-value limits through its
[`Limits`](https://docs.rs/tiff/0.9.1/tiff/decoder/struct.Limits.html) contract.
Those dependency limits bound one decode operation; the complete output still
scales with the number of accepted IFD pages. RITK detects allocation failure
while growing that output and returns an error.

## Failure behavior

Reading returns an error for:

- invalid TIFF/BigTIFF headers or directory structures;
- zero or overflowing page dimensions and RGB sample counts;
- unsupported color models;
- inconsistent page dimensions or a later page with another color model;
- decoded sample counts that disagree with declared geometry;
- decoder-limit, decompression, I/O, allocation, or image-construction failure.

The operation returns no partial image. The error identifies the page when the
failure occurs.

## Next

The [multi-page round-trip example](examples/tiff_roundtrip.md) proves pixel
and page-order equality with a difference panel, then makes the separate
geometry boundary visible instead of implying that TIFF preserves scanner
coordinates.
