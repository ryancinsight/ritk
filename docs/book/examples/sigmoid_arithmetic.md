# Example: Sigmoid and Arithmetic

The sigmoid maps an input intensity through

S(I) = (M - m) / (1 + exp(-(I - alpha) / beta)) + m.

Alpha is the midpoint and beta controls the transition width. RITK keeps the
output range explicit, which makes a preprocessing contract reproducible
across displays and downstream metrics.

![Sigmoid remapping in the complete processing pipeline](../figures/processing_pipeline.svg)

~~~rust,ignore
let normalized = SigmoidImageFilter::new(0.42, 0.10, 0.0, 1.0)
    .apply_native(&input, &backend)?;
~~~

Pointwise arithmetic filters use the same image boundary. Add, subtract,
multiply, and divide are binary image operations: both operands must have
compatible shapes and metadata. Use them to combine a corrected image with a
mask or to apply a calibrated scale after the nonlinear remap. Do not use a
display-only conversion as a substitute for a numeric arithmetic stage.

The pipeline figure makes the sigmoid effect readable by keeping the input and
sigmoid panels on the same [0, 1] display range. The output is visibly
different because the remap is data-derived, not because the renderer chooses
a new contrast window.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The sigmoid tests cover midpoint, monotonicity, bounded output, and the
degenerate beta step contract. Binary arithmetic tests cover shape mismatch and
value-semantic results.
