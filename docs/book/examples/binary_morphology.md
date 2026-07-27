# Example: Binary Morphology

Binary morphology operates on a foreground value and a structuring element.
For a cubic radius r, erosion retains a voxel only when its neighborhood is
foreground; dilation retains it when any neighborhood voxel is foreground.
Opening is erosion followed by dilation. Closing is dilation followed by
erosion.

![Binary opening and closing in the complete processing pipeline](../figures/processing_pipeline.svg)

~~~rust,ignore
let eroded = BinaryErodeFilter::new(1)
    .apply_native(&mask, &backend)?;
let opened = BinaryDilateFilter::new(1)
    .apply_native(&eroded, &backend)?;

let dilated = BinaryDilateFilter::new(1)
    .apply_native(&mask, &backend)?;
let closed = BinaryErodeFilter::new(1)
    .apply_native(&dilated, &backend)?;
~~~

The default foreground is 1.0 and out-of-bounds neighbors are background.
That boundary policy intentionally removes foreground touching the image edge
during erosion. If your segmentation uses another label, set it with
with_foreground and keep the background value consistent with the mask
contract.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The figure uses the same mask for both compositions and a fixed binary display
range. Tests cover radius zero identity, topology changes, border behavior,
foreground values, and native/generic parity.
