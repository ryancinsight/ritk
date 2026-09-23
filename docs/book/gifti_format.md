# GIFTI Surface Interchange

FreeSurfer's own surface files (`lh.white`, `lh.curv`, `lh.aparc.annot`) are
read by `ritk-parcellation::freesurfer`. GIFTI is the format other tools use to
exchange the same content, and `ritk-gifti` reads and writes it. The reference
is the GIFTI Surface Data Format 1.0 specification (14 January 2011).

## Layout

A GIFTI file is XML: a root `GIFTI` element holding optional file metadata, an
optional label table, and one or more `DataArray` elements. Each data array
declares:

| Attribute | Meaning |
|-----------|---------|
| `Intent` | What the values are: `NIFTI_INTENT_POINTSET` coordinates, `NIFTI_INTENT_TRIANGLE` vertex triplets, `NIFTI_INTENT_LABEL` keys into the label table, `NIFTI_INTENT_SHAPE` measurements, statistics |
| `DataType` | `NIFTI_TYPE_UINT8`, `NIFTI_TYPE_INT32`, or `NIFTI_TYPE_FLOAT32` |
| `Dimensionality`, `Dim0`… | The shape; node-based data puts the vertex count in `Dim0` |
| `ArrayIndexingOrder` | `RowMajorOrder` (last index fastest) or `ColumnMajorOrder` |
| `Encoding` | `ASCII`, `Base64Binary`, `GZipBase64Binary`, or `ExternalFileBinary` |
| `Endian` | Byte order of binary data |

The file-type extensions only name conventions: a `.surf.gii` holds one point
set and one triangle array, a `.func.gii` or `.shape.gii` one float per vertex,
and a `.label.gii` a label table plus integer keys per vertex.

```rust,ignore
use ritk_gifti::GiftiImage;

let surface = GiftiImage::read(std::fs::File::open("lh.pial.surf.gii")?)?.surface()?;
println!("{} vertices, {} triangles", surface.vertices.len(), surface.triangles.len());
```

## The encoding named gzip is zlib

`GZipBase64Binary` is base64 over a **zlib** stream, not a gzip file: the
specification says the data "is compressed using ZLIB" (section 5.0), and
nibabel inflates it with `zlib.decompress`. A reader that expects the gzip
header rejects every conforming file.

## What the reader checks

The reader treats every document as hostile input:

- Element nesting follows the DTD. A `Data` outside a `DataArray`, or a second
  root element, is an error. Elements the DTD does not define are skipped.
- A shape has one to six axes and at most 10⁸ values, checked before any
  payload is decoded.
- Each payload must decode to exactly the declared number of values. A zlib
  stream is inflated no further than one byte past that length, so a stream
  that expands beyond its shape is rejected without being inflated.
- `NumberOfDataArrays` must match the arrays present, and label colours must
  lie in `[0, 1]`.

Column-major arrays are kept as stored, and `DataArray::row_major` transposes
them. The writer always emits row-major, little-endian data, the forms every
reader accepts.

## What it does not do

`ExternalFileBinary` data is reported as unsupported. Resolving it needs the
directory of the XML file, and a reader of bytes does not have that path.
CIFTI, which embeds GIFTI-like arrays in NIfTI-2 containers, is out of scope.
