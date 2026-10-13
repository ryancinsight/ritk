# PNG Format Boundary

PNG is a practical import and export format for screenshots, microscopy
slices, QA artifacts, and simple test data. The native readers decode a single
image or stack a lexically ordered directory into a leading depth axis; the
native writer emits a single `[1, rows, cols]` slice as 8-bit grayscale.

~~~rust,ignore
use coeus_core::SequentialBackend;
use ritk_io::format::png::native::{PngReader, PngSeriesReader, PngWriter};
use ritk_io::{ImageReader, ImageWriter};

let slice = ImageReader::read(&PngReader::new(SequentialBackend), "slice.png")?;
let volume = ImageReader::read(&PngSeriesReader::new(SequentialBackend), "slices")?;
assert_eq!(slice.shape()[0], 1);
assert!(volume.shape()[0] >= 1);

// One slice in, one file out. A volume is rejected, not truncated.
ImageWriter::write(&PngWriter, "out.png", &slice)?;
~~~

Series stacking is lexical, so zero-pad slice names when numeric ordering is
required. PNG does not carry the same medical frame metadata as a volumetric
format; assign or validate spacing and direction before registration.

## Writing is windowed and lossy

PNG's grayscale format is 8-bit unsigned, so writing an `Image<f32>` requires
choosing a stored range. The writer maps the image's own `[min, max]` onto
`[0, 255]` and records nothing about it — PNG has no tag to record it in. A
reader gets back the rank ordering and the shape, not the original scale.
Callers who need the source values in the file's own units must pre-window the
image themselves. `write_png` requires `nz == 1`; use `write_png_volume` to
emit a `[depth, rows, cols]` image as a directory of slices.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native PNG import | Available | Covers single-slice decode and directory-series stacking. |
| [Windowing and Rescaling](examples/windowing_rescale.md) | Available | Shows the same intensity-boundary pattern on a CT fixture. |
