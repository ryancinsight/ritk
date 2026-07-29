# PNG Format Boundary

PNG is a practical import format for screenshots, microscopy slices, QA
artifacts, and simple test data. The native readers decode a single image or
stack a lexically ordered directory into a leading depth axis. The current
native contract is read-only.

~~~rust,ignore
use coeus_core::SequentialBackend;
use ritk_io::format::png::native::{PngReader, PngSeriesReader};
use ritk_io::ImageReader;

let slice = ImageReader::read(&PngReader::new(SequentialBackend), "slice.png")?;
let volume = ImageReader::read(&PngSeriesReader::new(SequentialBackend), "slices")?;
assert_eq!(slice.shape()[0], 1);
assert!(volume.shape()[0] >= 1);
~~~

Series stacking is lexical, so zero-pad slice names when numeric ordering is
required. PNG does not carry the same medical frame metadata as a volumetric
format; assign or validate spacing and direction before registration.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Native PNG import | Available | Covers single-slice decode and directory-series stacking. |
| [Windowing and Rescaling](examples/windowing_rescale.md) | Available | Shows the same intensity-boundary pattern on a CT fixture. |
