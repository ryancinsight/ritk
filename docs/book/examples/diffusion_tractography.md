# Example: Signal to Streamlines

This runnable example connects two independently verifiable stages: a
single-fiber diffusion-tensor signal is converted to an analytical Q-ball ODF,
then a separate curved unit-vector field is integrated into streamlines.

![Directional diffusion acquisition, analytical Q-ball ODF, and bounded streamline integration](../figures/diffusion_tractography.svg)

Read the numbered panels from left to right:

1. Blue points are the 48 actual unit gradient directions supplied to
   `GradientScheme`, all at \(b=1500\) s/mm². Two additional b0 measurements
   establish \(S_0\).
2. Orange points show the generated tensor signal against alignment with the
   known fiber axis. Stronger alignment causes greater attenuation. The blue
   polar shape is the ODF evaluated from RITK's fitted coefficients; the dashed
   red line is the independent analytical axis. The reported angular error is
   computed before the SVG is written.
3. Gray line segments are local direction samples, orange points are seeds,
   dashed curves bound the trackable field, and blue curves are Gaia polylines
   returned by `euler_tractography`. No curve contains the first proposal
   outside the dashed boundary.

The ODF panel and tractography panel intentionally use separate fields. This
prevents the figure from implying that one voxel's ODF is enough to form a
whole tract: tractography requires a spatial field of local orientations.

## Source and command

Source: `crates/ritk-diffusion/examples/book_diffusion_tractography.rs`

```text
cargo run -p ritk-diffusion --example book_diffusion_tractography -- \
  docs/book/figures/diffusion_tractography.svg
```

The example fails unless:

- a deterministic one-degree full-sphere search places the analytical Q-ball
  peak within two degrees of the known antipodal x axis;
- all five seeds produce streamlines;
- every emitted point remains inside the analytical vector-field domain; and
- the figure can be written without an unbounded or infallible allocation
  assumption in the library path.

This synthetic result verifies the stated numerical and geometry contracts.
It is not a claim of tractography accuracy on patient data.
