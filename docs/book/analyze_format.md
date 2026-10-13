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

That normalization deliberately differs from the NIfTI reader, which reports a
non-positive `pixdim` as a scoped spatial loss. NIfTI has `sform`/`qform`
alternatives and can still describe the volume when `pixdim` is unusable;
Analyze has no other spatial field, so a loss would leave the volume with no
geometry at all. Normalizing is the only outcome that returns an image, so the
divergence is a property of the formats rather than an inconsistency between
the readers.

The original header defines `originator` as ten history bytes, not a complete
world-space transform. RITK's writer uses the common five-`i16` convention and
stores rounded voxel coordinates. A physical origin therefore round-trips only
to the nearest voxel. The format has no direction matrix, so the RITK reader
returns identity direction and the writer **rejects a non-identity direction
before creating either file** rather than writing a dataset whose geometry
silently disagrees with the source. Analyze files cannot establish scanner-space
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

Writing rejects zero or oversized dimensions, storage/shape disagreement,
non-identity direction, and invalid spatial metadata before creating output.
I/O failures remain visible with their source chain.

## Checked conversion matrix

Each row maps one Analyze pair property to the observable RITK behavior, the
representable-or-lossy verdict, and the value test that pins it. Tests are
identified as `path::test` relative to `repos/ritk`; run them with
`cargo test -p ritk-analyze` (and `-p ritk-io` for the dispatch row).

| Property | RITK behavior | Typed verdict | Value test |
|---|---|---|---|
| Pair atomicity — preflight | Validation runs before either file is created; a rejected write leaves neither `.hdr` nor `.img`. | consistent pair, or neither destination | `crates/ritk-analyze/src/writer.rs::writer_rejects_invalid_input_before_creating_files`, `...::writer_rejects_a_non_identity_direction_before_creating_files` |
| Pair atomicity — commit order | The `.img` payload is written and flushed first; the `.hdr` is published last and is the commit marker. A failure after the payload leaves an inert orphan `.img`, never a header without its payload. | consistent pair, or orphan payload (recovery: re-run the write) | `crates/ritk-analyze/src/tests.rs::analyze_writer_publishes_the_header_only_after_the_payload` |
| Overwrite behavior | A write to an existing stem replaces both files; the `.img` is truncated to the new volume rather than appended to. | last write wins | `crates/ritk-analyze/src/tests.rs::analyze_writer_overwrites_an_existing_pair` |
| Scalar types — read | `u8` (2), `i16` (4), `i32` (8), `f32` (16), `f64` (64) decode to `f32`; `bitpix` must equal the datatype width; other codes are rejected before allocation. | narrowing to `f32`; `u16`/`u32`/`i64`/`u64` unrepresentable | `crates/ritk-analyze/src/tests.rs::analyze_reader_decodes_every_supported_scalar_with_scale`, `...::analyze_reader_rejects_invalid_geometry_and_bit_depth_before_allocation` |
| Scalar types — write | Emits `DT_FLOAT` (16) only. | `f32` → stored `f32` is exact | `crates/ritk-analyze/src/tests.rs::analyze_roundtrip_preserves_shape_spacing_origin_and_values` |
| Intensity scale | `funused1` (`0 → 1`) multiplies every decoded voxel; a non-finite value is rejected. | linear calibration folded into samples | `crates/ritk-analyze/src/tests.rs::analyze_reader_decodes_every_supported_scalar_with_scale`, `...::analyze_reader_rejects_non_finite_metadata_and_invalid_offsets` |
| Byte order — read | Little-endian only; a big-endian header is identified by name and rejected, not reported as corruption. | big-endian unrepresentable | `crates/ritk-analyze/src/tests.rs::analyze_reader_rejects_invalid_geometry_and_bit_depth_before_allocation` (asserts `"big-endian"`) |
| Byte order — write | Emits little-endian header fields and little-endian `f32` payload; output is byte-stable for a fixed logical image. | deterministic | `crates/ritk-analyze/src/tests.rs::analyze_writer_output_is_byte_stable_for_native_image` |
| Spacing | File `pixdim[1..3] = [sx, sy, sz]` is reversed to RITK tensor-axis `[sz, sy, sx]`; the writer reverses back. | exact for `f32`-representable spacing | `crates/ritk-analyze/src/tests.rs::analyze_writer_emits_pixdim_in_file_axis_order`, `...::analyze_roundtrip_preserves_shape_spacing_origin_and_values` |
| Spacing — legacy zero/negative | Non-finite spacing is rejected; `pixdim ≤ 0` falls back to unit spacing. The fallback is load-bearing: without it a legacy file panics in `Spacing::new`. | normalized to 1.0 mm | `crates/ritk-analyze/src/tests.rs::analyze_reader_falls_back_to_unit_spacing_for_non_positive_pixdim` |
| Origin | Reconstructed as `originator[i] × spacing[i]`; the writer rounds each world coordinate to the nearest `i16` voxel index and rejects an out-of-range result. | quantized to the nearest voxel | `crates/ritk-analyze/src/tests.rs::analyze_roundtrip_preserves_shape_spacing_origin_and_values`, `crates/ritk-analyze/src/writer.rs::writer_rejects_invalid_input_before_creating_files` |
| Direction | Analyze has no direction field. The reader returns identity; the writer rejects a non-identity direction before touching the file system. | non-identity unrepresentable (rejected, not dropped) | `crates/ritk-analyze/src/writer.rs::writer_rejects_a_non_identity_direction_before_creating_files` |
| Payload length and offset | `vox_offset` must be a finite non-negative whole byte count; the `.img` length must equal `vox_offset + voxel_count × width`. | exact | `crates/ritk-analyze/src/tests.rs::analyze_reader_requires_exact_payload_and_honors_offset`, `...::analyze_reader_rejects_non_finite_metadata_and_invalid_offsets` |
| Format identification | A paired NIfTI-1 (`ni1\0`) header is reported as NIfTI rather than decoded as Analyze. | routed to the NIfTI reader | `crates/ritk-analyze/src/tests.rs::analyze_reader_identifies_paired_nifti_header` |
| Dispatch contract | `ritk-io`'s `AnalyzeReader`/`AnalyzeWriter` round-trip a `[2, 3, 4]` volume under `SpatialFidelity::SpacingAndOrigin`. | shared contract parity | `crates/ritk-io/src/format/tests_native_readers.rs::native_analyze_writer_reader_contract_round_trips` |

A round trip alone cannot prove the axis order, because a self-consistent
transposition would pass it. `analyze_writer_emits_pixdim_in_file_axis_order`
therefore reads the raw header bytes at offsets 80, 84, and 88 and pins
`pixdim[1..3] = [3.75, 2.5, 1.25]` for core spacing `[1.25, 2.5, 3.75]`.

## Next

The [Analyze round-trip example](examples/analyze_roundtrip.md) uses a shared
display scale and a separate absolute-difference panel so visual similarity is
not mistaken for proof of equality.
