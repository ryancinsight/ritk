# MetaImage Format Boundary

MetaImage is a lightweight lossless boundary for header-driven volume
interchange. The native facade accepts both single-file MHA and header-plus-raw
MHD inputs and preserves validated shape, spacing, origin, and direction.

~~~rust,ignore
let image = ritk_io::read_image_native("volume.mha")?;
ritk_io::write_image_native("volume_copy.mha", &image)?;
~~~

Use MHA when one self-contained file is preferable. Use MHD with a raw payload
when an external pipeline already expects separate header and voxel files. A
round trip must compare shape and physical metadata in addition to voxel values.
Readers construct Coeus-backed images directly on the selected native backend;
writers extract host data only at the format boundary.

## Sample types

The `ElementType` header names the stored sample type, and `ritk-metaimage`
reads and writes all ten fixed-width numeric types a RITK sample can hold:

| `ElementType` | Stored sample | Exact reads | Cast reads |
|---|---|---|---|
| `MET_CHAR` | `i8` | `i8` and every wider signed type, `f32`, `f64` | any type |
| `MET_UCHAR` | `u8` | `u8` and every wider type | any type |
| `MET_SHORT` | `i16` | `i16`, `i32`, `i64`, `f32`, `f64` | any type |
| `MET_USHORT` | `u16` | `u16`, `u32`, `u64`, `i32`, `i64`, `f32`, `f64` | any type |
| `MET_INT` | `i32` | `i32`, `i64`, `f64` | any type |
| `MET_UINT` | `u32` | `u32`, `u64`, `i64`, `f64` | any type |
| `MET_LONG_LONG` | `i64` | `i64` | any type |
| `MET_ULONG_LONG` | `u64` | `u64` | any type |
| `MET_FLOAT` | `f32` | `f32`, `f64` | any type |
| `MET_DOUBLE` | `f64` | `f64` | any type |

`MET_LONG` and `MET_ULONG` read as `i32` and `u32`: MetaIO stores them in
four bytes on every platform (`MET_ValueTypeSize` in MetaIO's
`src/metaTypes.h`). The writer names those types `MET_INT` and `MET_UINT`.

`read_metaimage::<T, ..>` returns an image of the caller's sample type `T`
under a conversion policy (ADR 0053). The payload decodes in the stored type,
in the byte order `BinaryDataByteOrderMSB` selects, and is zlib-inflated first
when `CompressedData = True`; this holds for inline (`ElementDataFile = LOCAL`)
and detached (`.mhd` with a raw file) payloads alike. `Exact` returns the stored
values or a lossless widening and refuses a read that could change a value: a
`MET_INT` file read as `f32` fails, because binary32 holds integers exactly only
up to 2^24, and a `MET_LONG_LONG` file never passes through a float. `Cast`
converts and logs a warning when the stored type does not widen to `T`. The
`f32` surfaces of `ritk-io` read under `Cast`, the conversion they always
performed.

The reader streams the payload in bounded steps and requires it to hold exactly
the `DimSize` sample count, so a header that overstates the volume allocates no
more than the file supplies and trailing bytes are rejected.

`write_metaimage` stores the image's own sample type: the `ElementType` names
`T`, the payload is little-endian and uncompressed, and every sample round-trips
bit for bit, including signed zero.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native MetaImage round trip | Available | Demonstrates MHA and MHD/raw reads and writes through the unified facade. |
| [DICOM to NIfTI Conversion](examples/dicom_to_nifti.md) | Available | Shows the same image-boundary style used by format conversion workflows. |
