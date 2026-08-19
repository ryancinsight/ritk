# ritk-block-matching

Dependency-light block-matching displacement estimation for speckle tracking
and elastography.

The crate compares a fixed block with candidate blocks in a moving image using
zero-mean normalized cross-correlation and returns the displacement of the
best peak. Parabolic and cosine subpixel refinements are available. Inputs are
caller-owned flat row-major `[z, y, x]` sample buffers; the crate does not
depend on an image, tensor, or device backend.

```rust
use ritk_block_matching::{
    match_block, BlockMatchingConfig, SubpixelRefinement,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixed: Vec<f32> = (0..81)
        .map(|index| (index % 11) as f32 + index as f32 * 0.001)
        .collect();
    let moving = fixed.clone();
    let config = BlockMatchingConfig {
        block_radius: [0, 1, 1],
        search_radius: [0, 1, 1],
    };
    let displacement = match_block(
        &fixed,
        &moving,
        [1, 9, 9],
        [0, 4, 4],
        config,
        SubpixelRefinement::Parabolic,
    )?;
    assert!(displacement.peak_similarity > 0.99);
    Ok(())
}
```

Use [`MultiResolutionSearch`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/struct.MultiResolutionSearch.html)
when the caller owns a coarse-to-fine image pyramid. `match_pyramid` handles
one centre; `track_volume_pyramid` applies the same propagated-centre contract
to every valid block in a [`BlockGrid`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/struct.BlockGrid.html).
`track_volume_pyramid_regularized` then applies a configured Bayesian prior
using each finest-level peak as confidence, while preserving centres and peak
metadata. Use `track_volume_pyramid_diagnostics` (or its FFT counterpart) when
per-block coarse-to-fine centres and level peaks must be retained; skipped
blocks are represented by `None` diagnostics and `NAN` confidence.
`PyramidDisplacementField::validate` checks the four aligned public arrays and
retained level entries; use `try_as_field` when projecting manually assembled
or untrusted diagnostics into a `DisplacementField`. The infallible
`as_field` convenience retains the same validation and is intended for results
returned by the matcher.
`DisplacementPipeline::run_pyramid_with_diagnostics` retains the same raw
evidence while still applying configured strain and Bayesian stages to its
 returned field. `DisplacementPipeline::run_pyramid` integrates this batch path with
the pipeline's metric, refinement, strain-window, and Bayesian stages;
`run_owned_pyramid` is the convenience adapter for [`OwnedPyramid`] values
constructed by the crate's nearest-neighbour or min/max builders, and
`run_owned_pyramid_with_diagnostics` retains their raw level evidence. With the
optional `fft` feature, `match_pyramid_fft`, `track_volume_pyramid_fft`, and
`DisplacementPipeline`'s `PipelineMetric::Fft` mode run the same
propagated-centre pipeline through Apollo's explicit zero-padded linear NCC. Their outputs and per-level
coordinates are parity-tested against the direct path; zero padding is an FFT
work-buffer policy, not circular correlation or candidate evidence.

For untrusted block radii, use `BlockGrid::try_dense` instead of the
infallible `BlockGrid::dense`; execution paths validate all grid strides and
safely skip centres whose block extent would overflow or leave the image.

Two post-processing seams handle unreliable blocks, and they are complements
rather than alternatives. [`LeastSquaresDisplacementPrior`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/struct.LeastSquaresDisplacementPrior.html)
and [`BayesianDisplacementPrior`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/struct.BayesianDisplacementPrior.html)
*condition* every block, blending it toward a local least-squares slope or a
confidence-weighted prior. [`strain_window_filter`](https://docs.rs/ritk-block-matching/latest/ritk_block_matching/fn.strain_window_filter.html)
instead *rejects*: a block whose implied axial strain exceeds a plausibility
bound is replaced by interpolation between its nearest measured neighbours, and
everything else is returned untouched. Use the priors against measurement noise
and the filter against peak hopping, where a decorrelated block reports a
maximum from the wrong correlation lobe and is wrong by roughly a wavelength.
Blocks with no measured neighbour to draw on are reported rather than invented.

For acquisition-aware geometry, use
`BlockMatchingConfig::from_axial_autocorrelation` or
`BlockMatchingConfig::from_transducer_bandwidth`. Both derive an axial
half-length, map it explicitly onto the selected `[z, y, x]` axis, assign the
two transverse radii, and validate the resulting block/search geometry before
matching begins.

Use `DisplacementField::valid_mask(minimum_peak_similarity)` to validate the
parallel field arrays and select finite, confidence-qualified blocks. Use
`strain_from_displacement_filtered` when invalid blocks must be omitted from
finite differences; it leaves those output entries as `NAN` and scales across
skipped axial grid gaps. `PipelineStages::validate` checks manually assembled
priors, strain windows, and thresholds before matching. Set
`PipelineStages::minimum_peak_similarity` to use
that filtered estimator automatically in `DisplacementPipeline::run` and
`run_pyramid`. For manually assembled fields, use
`try_strain_from_displacement`, `BayesianDisplacementPrior::try_regularize`,
or `StrainWindowRegularizer::try_regularize`; these validate aligned arrays and
return errors instead of truncating or indexing malformed data. The infallible
helpers remain convenient for matcher-produced fields. Matcher outputs retain
`NAN` confidence and zero displacement for non-evaluable blocks; these APIs make
that convention explicit for strain or export consumers.

The algorithm follows the metric-image and displacement-calculator split from
ITKUltrasound and the sub-sample estimators described by Céspedes et al.
(1995). Boundary candidates are left unevaluated rather than padded or
silently clamped.
