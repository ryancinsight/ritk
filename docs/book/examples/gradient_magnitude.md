# Example: Gradient Magnitude

For a scalar image I on a regular grid, RITK estimates physical derivatives
with central differences and reports

|grad I| = sqrt((dI/dz)^2 + (dI/dy)^2 + (dI/dx)^2).

Spacing is part of the calculation. A two-voxel intensity change represents a
different physical slope when the voxel spacing changes.

![Gradient magnitude in the complete processing pipeline](../figures/processing_pipeline.svg)

~~~rust,ignore
let gradient = GradientMagnitudeFilter::new(*input.spacing())
    .apply_native(&input)?;
~~~

GradientMagnitudeFilter::unit() is useful only when the image has unit
spacing. For physical measurements, pass the image spacing explicitly. The
native implementation uses a zero-flux boundary stencil and preserves the
input geometry.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

The figure uses one data-derived upper bound for the gradient panel and prints
that bound below the image. This prevents a gradient map from looking
identical to the input merely because both were independently normalized.
Filter tests cover constant fields, linear fields with known spacing, boundary
behavior, and native/generic parity.
