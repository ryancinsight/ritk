# Image format dispatch migration

This guide accompanies the addition of MINC2 to the shared `ritk-io`
`ImageFormat` and the explicit output-dispatch API.

## Exhaustive matches

`ImageFormat` is now non-exhaustive. Downstream matches must handle the format
variants they use and include a wildcard arm for formats they do not handle.
The new `ImageFormat::Minc` variant recognizes `.mnc` and `.mnc2` paths and
routes MINC2 reads and writes through RITK's existing `ritk-minc` codec.

## Explicit output selection

`write_image_native` continues to infer file formats from path extensions.
Call `write_image_native_with_format` when the format is selected separately
from the path, or when the output is a directory, as with DICOM Secondary
Capture series:

```rust,ignore
write_image_native_with_format(output_directory, &image, ImageFormat::Dicom)?;
```

DICOM output is derived and does not copy patient or study metadata. PNG output
accepts one grayscale slice with finite integral samples from 0 through
65,535 and does not preserve physical-space metadata. JPEG output is lossy and
one-slice only. Other format-specific geometry limits are documented by the
`ritk-io` crate manual.
