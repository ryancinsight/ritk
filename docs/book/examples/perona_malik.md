# Example: Perona–Malik Diffusion

Perona–Malik diffusion evolves

dI/dt = div(c(|grad I|) grad I),

where the conductance c is large in homogeneous regions and small across
strong gradients. RITK provides exponential and quadratic conductance
strategies and an explicit Euler configuration.

![Perona–Malik diffusion and its absolute change map](../figures/processing_pipeline.svg)

~~~rust,ignore
let config = DiffusionConfig {
    num_iterations: 12,
    time_step: 0.0625,
    conductance: 0.08,
    function: ConductanceFunction::Exponential,
};
let diffused = config.apply_native(&input, &backend)?;
~~~

For a three-dimensional unit grid, the explicit time step must satisfy the
stability bound dt <= 1/6. The example uses 1/16. Conductance is expressed in
the same intensity-gradient units as the image; it is not a display contrast
parameter.

The figure uses a shared [0, 1] scale for input and output and adds an
absolute change panel. That diagnostic is required when visual contrast alone
cannot show whether a denoiser changed the data.

## Source and verification

Source: crates/ritk-filter/examples/book_processing_pipeline.rs

~~~text
cargo run -p ritk-filter --example book_processing_pipeline -- \
  docs/book/figures/processing_pipeline.svg
~~~

Tests cover constant-field invariance, conductance selection, stability-sized
steps, metadata preservation, and native/generic parity.
