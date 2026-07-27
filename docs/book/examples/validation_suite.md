# Example: Registration Validation

Validation is a separate stage from optimization. A registration can improve a
metric while violating geometry, producing an empty resample, or moving a
label boundary in the wrong direction. A useful report combines:

- shape and physical-frame checks;
- metric values before and after;
- overlap measures when labels exist; and
- convergence state and iteration budget.

The CT/MR registration example demonstrates the visual part of this contract:
it renders identity and registered overlays with red/green fringes and a
data-derived MR resampling-change map.

![Registration validation output](../figures/ct_mri_registration.svg)

For a label-space check, use the statistics facade after resampling both label
maps onto one grid:

~~~rust,ignore
let dice = ritk_statistics::dice_coefficient(&fixed_labels, &moving_labels)?;
let hausdorff = ritk_statistics::hausdorff_distance(&fixed_labels, &moving_labels)?;
~~~

The exact statistics signature depends on the label representation, so the
validation layer must keep the conversion at the boundary and report the
input shapes with the metric values. Never accept only an is_ok result: record
the value, units, reference frame, and threshold used for the decision.

## Source and verification

The visual source is crates/ritk-registration/examples/book_registration.rs.

~~~text
cargo run -p ritk-registration --example book_registration -- \
  docs/book/figures/ct_mri_registration.svg
~~~

The registration example requires normalized mutual information to improve from
identity to the dataset transform, preserves the full CT grid, and reports
maximum and mean absolute MR resampling change. The registration and
statistics test suites provide the numerical oracles; the figures are checked
for correct labels, shared display conventions, and visible pre/post change.
