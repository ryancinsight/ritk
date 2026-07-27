# Example: Curvature Flow

Curvature flow evolves a scalar field according to mean curvature. Unlike
ordinary isotropic diffusion, the update depends on local surface geometry and
is intended to smooth small-scale boundary roughness while retaining larger
features for longer.

![Curvature flow and its absolute change map](../figures/processing_pipeline.svg)

~~~rust,ignore
let filtered = CurvatureFlowImageFilter::new(CurvatureFlowConfig {
    num_iterations: 5,
    time_step: 0.0625,
})
.apply_native(&input, &backend)?;
~~~

The explicit Euler stability bound for a three-dimensional unit grid is
dt <= 1/6. RITK uses double-precision stencil arithmetic internally for the
curvature numerator and writes the result back to the image scalar type.
Spacing is read from the image metadata.

The complete pipeline source runs five steps and renders the result beside an
absolute input-to-output change panel. The change panel is not a second
contrast-adjusted image; it is a direct difference diagnostic.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

Curvature-flow tests cover constant images, stability-sized steps, boundary
stencils, iteration behavior, and native/generic parity.
