# JPEG Format Boundary

JPEG support occupies a different niche from NIfTI, NRRD, or MetaImage: it is primarily a visualization and interchange boundary, not a quantitatively exact archival format. `ritk-io` re-exports `JpegColorReader` plus native `JpegReader` and `JpegWriter` adapters from `ritk-jpeg`, allowing scalar images to be read into or written from the standard `ritk-image::Image` contract. Because JPEG is lossy, round trips are validated with tolerances rather than bitwise equality, and the expected use cases are overlays, previews, report figures, and lightweight exports.

The Atlas integration story is still important. Once decoded, JPEG data becomes an ordinary Coeus-backed image that can flow through the same intensity, filtering, and registration APIs as any other input. The format layer owns compression artifacts and file handling; Coeus, Leto, and Moirai only come into play after the pixels have crossed the boundary. For that reason, JPEG is best used at the edges of a workflow, while authoritative computation stays in lossless in-memory or on-disk representations.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| Dedicated JPEG read/write walkthrough | Planned | Show tolerant round-trip expectations and recommend JPEG for previews rather than metric-sensitive pipelines. |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | Representative visualization workflow where compressed export is acceptable after alignment is complete. |
