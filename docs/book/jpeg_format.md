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

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native JPEG read/write boundary | Available | Uses path inference and toleranced pixel validation. |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | Representative visualization workflow where compressed export is acceptable after alignment. |
