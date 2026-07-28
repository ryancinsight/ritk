# Example: Grayscale Morphology

Grayscale opening is a local minimum followed by a local maximum. It removes
bright protrusions smaller than the structuring element while retaining the
image geometry. Grayscale closing performs the dual maximum-then-minimum
operation and fills small dark holes.

![Grayscale opening in the complete processing pipeline](../figures/processing_pipeline.svg)

~~~rust,ignore
let opened = GrayscaleOpeningFilter::new(2)
    .apply_native(&sigmoid, &backend)?;
let closed = GrayscaleClosingFilter::new(2)
    .apply_native(&sigmoid, &backend)?;
~~~

The radius is measured in voxels, not physical units. Choose it from the
feature size you intend to remove and account for anisotropic spacing before
using the filter on a clinical volume. Both operations use replicate padding
at the boundary, so their behavior differs from binary erosion's
background-outside policy.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The figure renders the opening stage on [0, 1]; the closing call above follows
the same image contract and can be inspected with the same display helper. The
opening result therefore shows local intensity removal directly. Tests cover
radius zero identity, monotonicity, idempotence, safe-border behavior, and
native/generic parity.
