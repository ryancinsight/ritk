# PNG Format Boundary

PNG support in ritk is intentionally narrow and practical: decode single images into a scalar or color `Image`, or stack a lexically ordered directory of slices into a `[depth, row, col]` volume. The `ritk-io::format::png` facade exposes `PngReader` and `PngSeriesReader` over the native `ritk-png` crate, making PNG a convenient ingress format for screenshots, microscopy slices, QA artifacts, and simple test data. The native contract currently covers reading only; the unified writer path deliberately rejects PNG output so the rest of the pipeline does not promise a capability the format layer does not implement.

Atlas integration stays the same as for medical formats: the reader constructs a Coeus-backed `ritk-image::Image`, and downstream filters see only that image boundary, not PNG-specific details. In practice, PNG is best treated as a lightweight import format before windowing, thresholding, edge detection, or registration preprocessing. That keeps the format boundary thin while letting Coeus and Moirai handle compute deeper in the pipeline.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Standalone PNG import walkthrough | Planned | Cover single-slice decode and directory-series stacking through the native PNG readers. |
| [Windowing and Rescaling](examples/windowing_rescale.md) | Planned | Common follow-on workflow after loading PNG slices or QA images into a scalar volume. |
