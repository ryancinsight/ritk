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

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native MetaImage round trip | Available | Demonstrates MHA and MHD/raw reads and writes through the unified facade. |
| [DICOM to NIfTI Conversion](examples/dicom_to_nifti.md) | Available | Shows the same image-boundary style used by format conversion workflows. |
