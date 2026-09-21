# JPEG Format Boundary

JPEG is a visualization and interchange boundary, not a quantitatively exact
archival format. It is appropriate for previews, overlays, reports, and
lightweight exports after metric-sensitive computation has completed.

The native facade infers the format from the path:

~~~rust,ignore
let preview = ritk_io::read_image_native("aligned_preview.jpg")?;
ritk_io::write_image_native("aligned_preview_copy.jpg", &preview)?;
~~~

JPEG quantization changes intensities, so a round trip must use a bounded error
or perceptual check rather than bitwise equality. Keep NIfTI, NRRD, or
MetaImage as the quantitative source for registration and metric evaluation.
After decoding, the result is an ordinary RITK image and can enter the same
filter pipeline as any other input.

The file reader maps the JPEG frame's full encoded range to 0–255 with nearest
integer rounding before grayscale conversion or RGB image construction. This
display policy applies equally to low-precision lossless and 12-bit DCT input.
DICOM decoding uses the clinical path instead: it preserves full integer sample
values and requires the frame precision to match BitsStored.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native JPEG read/write boundary | Available | Uses path inference and toleranced pixel validation. |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Representative visualization workflow where compressed export is acceptable after alignment is complete. |
