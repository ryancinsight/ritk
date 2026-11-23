# MetaImage Format Boundary

The native reader loads scalar MHA images and MHD headers with a single
external payload into a three-dimensional floating-point image. It carries
the shape and voxel-to-world geometry into the compute image.

~~~rust,ignore
let image = ritk_io::read_image_native("volume.mha")?;
ritk_io::write_image_native("volume_copy.mha", &image)?;
~~~

The native writer emits a single file with an inline payload regardless of
the output suffix. Use an `.mha` suffix for interoperability. A
round trip must compare shape and physical metadata in addition to voxel values.
Readers construct Coeus-backed images directly on the selected native backend;
writers extract host data only at the format boundary.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native MetaImage round trip | Available | Reads MHA and MHD/raw through the unified facade and writes single-file MHA. |
| [DICOM to NIfTI Conversion](examples/dicom_to_nifti.md) | Available | Shows the same image-boundary style used by format conversion workflows. |

## Conversion inventory

The table distinguishes the compute API in `ritk-metaimage` from the stored-volume
conversion contract. An exact adapter must resolve each loss before opening the
destination. The file fields follow the [MetaIO specification](https://docs.itk.org/en/latest/learn/metaio.html).

| Field | Current compute path | Exact conversion requirement and existing evidence |
| --- | --- | --- |
| `NDims`, `DimSize` | Reads ranks 2 and 3. A 2D image gains one slice. File order is X, Y, Z and the tensor shape is Z, Y, X. | Preserve the input rank or report a rank change. Check every sample at its X-fastest index. Reader tests cover shape and payload order. |
| `ElementType` | Reads UCHAR, SHORT, USHORT, INT, UINT, FLOAT and DOUBLE. The compute image and writer use binary32. | Carry the stored scalar type and its bits. For example, 16,777,217 cannot survive an INT → binary32 → INT conversion. |
| Byte order and compression | Reads raw or zlib payloads and both MetaIO byte-order field names. Writes little-endian uncompressed samples. | Packaging may change when decoded bits match. The byte-order test covers four signed integers in both byte orders and all four inline/detached and raw/zlib combinations. |
| `ElementDataFile` | Reads `LOCAL` or one named payload. All writes emit `LOCAL`. | Distinguish detached packages, file lists, and offsets. Retain or reject unsupported layouts explicitly. The external-raw reader test checks the decoded voxel sequence. |
| `Offset`, `Position`, `ElementSpacing`, `TransformMatrix` | Reads a physical origin, spacing and file-axis directions. The writer serializes the direction of each X, Y, Z axis in order. | Preserve the voxel-to-world map. Qualify alternate geometry keys and units before treating them as equivalent. The nonidentity matrix writer test and Python SimpleITK tests check axis order in both directions. |
| Intensity slope and offset | The compute reader decodes numeric samples. It does not retain MetaIO intensity conversion fields. | Preserve calibration separately from stored samples, or return a typed loss naming the field. A nonzero intercept cannot be represented by an identity calibration. |
| Channels, axis semantics and application metadata | A scalar image has no slot for additional channel identities, acquisition axes, or arbitrary header fields. | Require an explicit mapping or scoped loss for each field. A matching payload length alone cannot establish semantic equivalence. |

The [reader tests](../../crates/ritk-metaimage/src/tests/reader.rs),
[byte-order tests](../../crates/ritk-metaimage/src/tests/byte_order.rs),
[writer tests](../../crates/ritk-metaimage/src/tests/writer.rs), and
[SimpleITK comparisons](../../crates/ritk-python/tests/test_sitk_real_data_parity.py)
cover the current compute interface. The stored-volume adapter remains open.
