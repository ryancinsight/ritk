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

`ritk_metaimage::read_metaimage` takes an `ImageReadBudget`. The `DimSize`
product and `ElementType` imply the inflated payload and the `f32` output, and
both are checked against the decoded-byte ceiling before a zlib payload is
inflated; the encoded payload is read through a reader bounded by the
encoded-byte ceiling. A size above a ceiling returns `MetaImageError::Budget`.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native MetaImage round trip | Available | Demonstrates MHA and MHD/raw reads and writes through the unified facade. |
| [DICOM to NIfTI Conversion](examples/dicom_to_nifti.md) | Available | Shows the same image-boundary style used by format conversion workflows. |
