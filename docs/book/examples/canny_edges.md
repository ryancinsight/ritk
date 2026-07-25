# Example: Canny Edge Detection

The filter gallery runs the complete Canny pipeline on the same deterministic
phantom as the Gaussian example. `CannyEdgeDetector::apply_native` performs
Gaussian pre-smoothing, central-difference gradients, continuous-direction
non-maximum suppression, and 26-connected hysteresis.

![Canny edge map](../figures/filter_gallery.svg)

## Source and command

Source: `crates/ritk-filter/examples/book_filter_gallery.rs`

```text
cargo run -p ritk-filter --example book_filter_gallery -- \
  docs/book/figures/filter_gallery.svg
```

The output is binary (`1.0` at retained edges, `0.0` elsewhere) and preserves
the input geometry. Threshold behavior is covered by the package's value
semantic tests; the SVG makes the connected edge structure inspectable.
