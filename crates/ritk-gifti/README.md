# ritk-gifti

Reads and writes GIFTI, the XML interchange format for data on surface meshes:
triangle surfaces (`.surf.gii`), per-vertex measurements (`.func.gii`,
`.shape.gii`, `.time.gii`), and parcellations with their label table
(`.label.gii`).

A GIFTI file is a sequence of *data arrays*, each with an intent that says what
it holds (coordinates, triangles, labels, statistics), a numeric type, a shape,
and an encoding: whitespace-separated ASCII, base64, or base64 over a zlib
stream. This crate decodes all three, in either byte order and either indexing
order, and treats every file as hostile input: shapes are bounded, decoded
lengths must match the declared shape exactly, and failures are typed
`GiftiError`s rather than panics.

```rust
use ritk_gifti::{ArrayData, DataArray, DataEncoding, GiftiImage, Intent, MetaData};

let points = DataArray::new(
    Intent::PointSet,
    vec![3, 3],
    ArrayData::Float32(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].into()),
)?;
let triangles = DataArray::new(
    Intent::Triangle,
    vec![1, 3],
    ArrayData::Int32(vec![0, 1, 2].into()),
)?;
let image = GiftiImage::new(MetaData::default(), Vec::new(), vec![points, triangles])?;

let mut xml = Vec::new();
image.write(&mut xml, DataEncoding::GZipBase64Binary)?;
let surface = GiftiImage::read(xml.as_slice())?.surface()?;
assert_eq!(surface.triangles, vec![[0, 1, 2]]);
# Ok::<(), ritk_gifti::GiftiError>(())
```

Data stored in an external file (`ExternalFileBinary`) is reported as
unsupported: resolving it needs the path of the XML file, which a reader of
bytes does not have.

Specification: GIFTI Surface Data Format, version 1.0 (14 January 2011),
<https://www.nitrc.org/projects/gifti>.

License: MIT OR Apache-2.0.
