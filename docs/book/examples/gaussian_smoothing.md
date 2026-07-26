# Example: Gaussian Smoothing

This example is the Gaussian stage of the shared filter gallery. It constructs
a deterministic scalar phantom, wraps it in `ritk_image::Image<f32, _, 3>`,
and applies `GaussianFilter::apply_native` with a physical sigma of 2.0.

The figure compares the input with the smoothed image and the Canny edge map
that consumes the same image boundary:

![Gaussian smoothing result](../figures/filter_gallery.svg)

## Source and command

Source: `crates/ritk-filter/examples/book_filter_gallery.rs`

```text
cargo run -p ritk-filter --example book_filter_gallery -- \
  docs/book/figures/filter_gallery.svg
```

The image shape and spatial metadata are unchanged by the filter. The package
tests cover native/generic parity; the committed SVG is the visual output of
this example's figure contract. Rerun the command after changing the example
to regenerate the artifact from the computed values.
