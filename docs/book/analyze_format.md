# Analyze 7.5 Format Boundary

Analyze 7.5 stores one logical dataset in two files with the same stem:

```text
brain.hdr   348-byte binary header
brain.img   contiguous binary voxel payload
```

The original format description is the
[Analyze 7.5 File Format specification](https://analyzedirect.com/documents/AD_AnalyzeImage75_File_Format.pdf),
especially the header organization and image-dimension definitions on pages
3–8. The header gives dimensions, scalar type, bit depth, spacing, and payload
offset. The payload contains no delimiters: its expected length follows from
the dimension product and bits per voxel.

## RITK's supported subset

RITK deliberately reads a narrow, unambiguous subset:

| Property | Accepted contract |
|---|---|
| Byte order | little-endian |
| Logical shape | one 3-D volume |
| Header | exactly 348 bytes |
| Scalar types | `u8`, `i16`, `i32`, `f32`, or `f64` |
| Returned voxels | `f32` |
| Payload offset | finite, non-negative whole-byte offset |
| Payload length | exactly offset plus declared voxel bytes |

Big-endian files, four-dimensional series, complex values, RGB values, and
negative per-image offsets are rejected with contextual errors. Rejecting an
unsupported variant is preferable to decoding it with the wrong byte order or
geometry.

The extensions are not sufficient format identification. A paired NIfTI-1
dataset uses `ni1\0` at header bytes 344–347 and commonly includes the four-byte
extension indicator after the shared 348-byte header. RITK reports that case as
paired NIfTI instead of decoding NIfTI affine fields as Analyze history fields.
Use the `.nii` single-file form with RITK's NIfTI reader. The
[NIfTI-1 FAQ](https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html)
documents the shared pair layout and extension indicator; the
[SimpleITK I/O list](https://simpleitk.readthedocs.io/en/main/IO.html)
documents `NiftiImageIO` as the standard handler for these extensions.

The public API is backend-bound but format behavior is shared:

```rust,ignore
use coeus_core::SequentialBackend;
use ritk_analyze::{read_analyze, write_analyze};

let backend = SequentialBackend;
let image = read_analyze("brain.hdr", &backend)?;
write_analyze("copy.hdr", &image, &backend)?;
```

Passing `brain.img` to `read_analyze` is equivalent; the reader derives both
sibling paths from the stem.

## Axis order and byte order

The file describes dimensions as `[x, y, z]`. X varies fastest in the payload:

```text
file_index(x, y, z) = x + nx·y + nx·ny·z
```

RITK stores a three-dimensional image with shape `[z, y, x]` and the same
X-fastest flat order:

```text
ritk_index(z, y, x) = z·ny·nx + y·nx + x
```

The flat sequences are identical, so reading does not transpose or copy an
intermediate volume. The header spacing does require a semantic reorder:
file `[sx, sy, sz]` becomes RITK tensor-axis spacing `[sz, sy, sx]`.

## Scalar conversion and scaling

`datatype` selects the stored scalar and `bitpix` must match it. RITK checks
that pair before calculating payload bytes. Each stored value is converted to
`f32`; integer values outside binary32's exact integer range and finite `f64`
values can round under that explicit output contract.

The historical `funused1` field is used by several Analyze-derived writers as
an intensity scale. RITK applies a finite nonzero scale after scalar decoding;
zero means one. This convention is not uniform across every Analyze variant,
so a foreign pipeline should verify representative values rather than infer
calibration from the filename.

## Spatial metadata limits

Analyze `pixdim[1..3]` carries voxel spacing. Non-finite values are rejected;
legacy zero or negative spacing is normalized to unit spacing for compatibility.

The original header defines `originator` as ten history bytes, not a complete
world-space transform. RITK's writer uses the common five-`i16` convention and
stores rounded voxel coordinates. A physical origin therefore round-trips only
to the nearest voxel. The format has no direction matrix, so the RITK reader
returns identity direction. Analyze files cannot establish scanner-space
orientation as precisely as NIfTI, DICOM, NRRD, or MGH.

The [NIfTI-1 rationale](https://nifti.nimh.nih.gov/dfwg/presentations/nifti-1-rationale.html)
explains why NIfTI retained the 348-byte pair layout while adding explicit
coordinate-system semantics. Because both formats can use `.hdr` and `.img`,
select the reader from an authoritative source; extensions alone do not prove
which format is present.

## Bounded decoding and writing

Before allocating output, the reader validates signed dimensions, checked
voxel and byte products, datatype/bit-depth agreement, finite metadata, offset,
and exact file length. It then fallibly reserves the final `Vec<f32>` and
streams conversion through an 8 KiB fixed buffer. Peak decoder-owned storage is
therefore the returned `f32` volume plus constant scratch, not the complete
encoded payload plus the returned volume.

The writer validates dimensions, value count, checked byte size, finite spacing
representable in the header's `f32` fields, and origin voxel coordinates within
the format's `i16` range before creating either file. Header spacing can round
from RITK's `f64` metadata to `f32`. The writer streams little-endian voxel
bytes through an 8 KiB buffer and publishes the header after the payload
completes; it does not construct a second volume-sized byte vector.

## Failure behavior

Reading returns no partial image when any contract fails. Errors identify:

- invalid or unsupported dimensionality;
- unsupported endianness or scalar type;
- paired NIfTI presented through the ambiguous `.hdr`/`.img` extensions;
- mismatched `datatype` and `bitpix`;
- non-finite spacing, scaling, or offset;
- offset and byte-count overflow;
- truncated or trailing payload data;
- allocation, seek, read, or image-construction failure.

Writing rejects zero or oversized dimensions, storage/shape disagreement, and
invalid spatial metadata before creating output. I/O failures remain visible
with their source chain.

## Next

The [Analyze round-trip example](examples/analyze_roundtrip.md) uses a shared
display scale and a separate absolute-difference panel so visual similarity is
not mistaken for proof of equality.
