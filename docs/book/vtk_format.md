# VTK Format Boundary

The VTK boundary supports scalar image handoff and surface-oriented output.
For scalar image paths, use the same native facade as the other lossless
formats:

~~~rust,ignore
let image = ritk_io::read_image_native("volume.vtk")?;
ritk_io::write_image_native("volume_copy.vtk", &image)?;
~~~

Surface output uses explicit mesh and polydata writer APIs exported by
ritk-io, including write_mesh_as_vtk and the OBJ, PLY, STL, VTP, and glTF
helpers. A surface writer does not infer image spacing from a raw point list;
construct the mesh in the intended physical frame before exporting it.

## Direction-aware volume handoff

`ritk-snap::LoadedVolume` uses `[depth, row, column]` axes and keeps its
direction cosine matrix beside the decoded samples. The VTK boundary converts
that contract to a zero-copy [`ritk_vtk::VtkImageVolume`] in VTK order
`[x, y, z] = [column, row, depth]`:

```rust,ignore
let volume = ritk_vtk::VtkImageVolume::try_from(&loaded_volume)?;
assert_eq!(volume.dimensions(), [columns, rows, depth]);
let samples = volume.scalars();
```

The carrier preserves origin, spacing, direction, channel count and
x-fastest interleaved samples while sharing the source allocation. Call
`to_vtk_image_data` only at an existing serializer or filter boundary that
requires VTK's owned `Vec<f32>` attribute arrays; that operation is explicit
and retains the direction matrix on the carrier.

## Legacy structured-points scalar types

The legacy structured-points reader keeps the type the `SCALARS` line declares
and the writer stores the sample type of the image it is given, so a label map
or a CT written as `i16` reads back as `i16` and a `f64` image reads back as
`f64`. Binary data is big-endian. The names and widths are those of
`vtkDataReader::ReadArray` in Kitware/VTK `IO/Legacy/vtkDataReader.cxx`:

| `SCALARS` type name | Sample type | Bytes |
| --- | --- | --- |
| `unsigned_char` | `u8` | 1 |
| `char`, `signed_char` | `i8` | 1 |
| `unsigned_short` | `u16` | 2 |
| `short` | `i16` | 2 |
| `unsigned_int` | `u32` | 4 |
| `int`, `vtkidtype` | `i32` | 4 |
| `vtktypeuint64` | `u64` | 8 |
| `vtktypeint64` | `i64` | 8 |
| `float` | `f32` | 4 |
| `double` | `f64` | 8 |

The reader accepts both `char` and `signed_char` for `i8`, as `vtkDataReader`
does; the writer emits `char`. `vtkCharArray` holds a C `char`, whose
signedness the C standard leaves to the implementation, so reading `char` as
`i8` is a recorded choice, not a fact of the format. `vtktypeint64` and
`vtktypeuint64` are the 64-bit names `vtkDataReader` reads. `vtkidtype` reads
as `i32`: `vtkDataReader` reads that name as 4-byte big-endian `int` and
widens each value into a `vtkIdTypeArray`, and `vtkDataWriter` writes every
`vtkIdTypeArray` as `int` data. `bit` (eight samples to a byte) and `long` and
`unsigned_long` are refused by name: the binary width of `long` is that of the
writing machine's C `long`, 4 or 8 bytes, and the file does not record it.
`vtkDataWriter` still writes both names (`IO/Legacy/vtkDataWriter.cxx`, the
`VTK_LONG` and `VTK_UNSIGNED_LONG` cases of `WriteArray`), and `vtkDataReader`
reads them with `sizeof(long)`-wide values. A `SCALARS` array with more than
one component is refused; the structured-points reader returns one scalar per
point.

`ritk_vtk::read_vtk` takes the sample type of the returned image and a
conversion. `Exact` accepts the stored type or one it widens to, so an `int`
file reads as `i32` or `f64` and is refused as `f32`; `Cast` converts with the
primitive `as` cast and warns when a value may change. `ritk_io`'s `f32`
surfaces read under `Cast` until the dispatch reads in the caller's type.

XML data arrays (`.vti`, `.vtp`, `.vtu`) decode in their declared `type`
(`Int8` through `Float64`, ASCII or raw appended) and convert to the `f32`
attribute arrays of the VTK data model under `Cast`. Inline base64
(`format="binary"`), base64 appended data, and compressed appended data are not
implemented and are refused by name; a malformed token or a count that
contradicts `NumberOfTuples` is an error, not a dropped value.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native VTK image boundary | Available | Uses the unified image facade for supported VTK image paths. |
| Mesh and polydata export boundary | Available | Uses explicit mesh and polydata writers for surface handoff. |
