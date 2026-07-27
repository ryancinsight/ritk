# Example: A Complete Processing Pipeline

This example is the reference workflow for the filter chapters. It creates one
deterministic three-dimensional scalar image, applies several real RITK native
filters, and renders every result with an explicit display range.

![RITK processing pipeline showing intensity remapping, thresholding, gradients, morphology, diffusion, and a change map](../figures/processing_pipeline.svg)

Read the panels left-to-right, then continue on the next row:

1. the noisy scalar input;
2. a bounded sigmoid remap;
3. threshold suppression;
4. physical gradient magnitude;
5. binary opening;
6. binary closing;
7. grayscale opening;
8. Perona–Malik diffusion;
9. the absolute diffusion change;
10. curvature flow;
11. the absolute curvature change; and
12. the contract summary.

The input is a smooth phantom with two deterministic local perturbations,
making the diffusion change visible as spatial structures rather than a
directional stripe or fine-grain noise pattern. The input,
sigmoid, grayscale, and diffusion panels share [0, 1]. The binary
panels use the same range because their values are exactly 0 or 1. Gradient
and change panels use their own data-derived upper bounds, which are printed
under the panel. The phantom has three identical depth slices and the figure
shows the center slice, so radius-one 3-D morphology and diffusion operate on
a genuine volume rather than a depth-one edge case. This avoids the common
documentation error where every output is independently contrast-stretched
and therefore appears unchanged.

## Source and command

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The example uses Image::from_flat_on with [depth, row, column] shape and unit
physical spacing. Every filter receives the same image boundary and the native
SequentialBackend; no plotting library supplies the processed values.

## Reuse the individual stages

The core calls are:

~~~rust,ignore
let remapped = SigmoidImageFilter::new(0.42, 0.10, 0.0, 1.0)
    .apply_native(&input, &backend)?;
let mask = BinaryThresholdImageFilter::new(0.62, 1.0, 1.0, 0.0)
    .apply_native(&remapped, &backend)?;
let edges = GradientMagnitudeFilter::unit().apply_native(&input)?;
let denoised = DiffusionConfig {
    num_iterations: 12,
    time_step: 0.0625,
    conductance: 0.08,
    ..DiffusionConfig::default()
}
.apply_native(&input, &backend)?;
~~~

Binary opening and closing are explicit compositions in this example:
erode → dilate and dilate → erode. That makes the structuring-element
semantics visible and lets a caller substitute another radius or foreground
value without changing the image contract.

## Verification

- The example checks image construction and every native filter result through
  Result; no stage is replaced by a display-only approximation.
- All panels are generated from the same source and use fixed, labeled scales.
- The change panel is computed from the input and diffusion arrays, so it
  exposes whether the filter actually changed the data.
- Filter-specific algebraic and native-path tests remain the authoritative
  correctness checks; the SVG is the visual check for structure and readability.
