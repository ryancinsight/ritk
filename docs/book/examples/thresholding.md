# Example: Thresholding

Thresholding changes voxel values but not the image frame. RITK exposes a
retaining threshold and a binary indicator:

- ThresholdImageFilter::outside(lower, upper, outside) keeps values in the
  closed interval and replaces values outside it.
- BinaryThresholdImageFilter::new(lower, upper, foreground, background)
  writes the foreground value inside the interval and background elsewhere.

The complete pipeline figure shows both behavior classes. The threshold panel
is the suppression stage; the binary morphology panels consume a binary
indicator made from the same sigmoid output.

![Thresholding in the complete processing pipeline](../figures/processing_pipeline.svg)

~~~rust,ignore
let retained = ThresholdImageFilter::outside(0.0, 0.58, 0.0)
    .apply_native(&sigmoid, &backend)?;
let mask = BinaryThresholdImageFilter::new(0.62, 1.0, 1.0, 0.0)
    .apply_native(&sigmoid, &backend)?;
assert_eq!(retained.shape(), sigmoid.shape());
assert_eq!(mask.shape(), sigmoid.shape());
~~~

The interval is inclusive for the binary filter. That boundary matters for
label maps and should be tested with values exactly at both endpoints. Values
outside the interval are not clipped into the interval; they become the
configured background.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The native filter tests cover endpoint inclusion, outside suppression, shape
preservation, and native/generic parity. The figure verifies that the
foreground topology is visible and that thresholding is not an independently
contrast-stretched display artifact.
