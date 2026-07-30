# ADR 0015 — Fallible temporal synchronization

- Status: Accepted
- Date: 2026-07-30
- Board item: [SAFE-683-01](../../backlog.md#safe-683-01-major---make-temporal-synchronization-safe-dimensionally-correct-allocation-efficient-and-teachable)

## Context

The public temporal synchronizer accepts arbitrary `TemporalSyncConfig` field
values and does not reject non-finite signal samples. A NaN can therefore reach
`partial_cmp(...).expect(...)` during peak selection and panic. Zero-variance
signals are unidentifiable by normalized cross-correlation, but two identical
constant signals are reported as perfectly synchronized.

The configured `min_correlation` is not used. Instead, a separate quality
function applies hard-coded correlation thresholds and compares an
intensity-residual magnitude with `frame_spacing`, which is measured in
seconds. The public metric names and documentation then report those intensity
residuals as timing errors. Residual RMS is also divided by the complete signal
length even when only a smaller shifted overlap contributes.

`synchronize` allocates complete lag and correlation arrays although it retains
only the maximum and its two neighbors. Callers that need the complete profile
for diagnostics cannot request it through the public API.

Direct temporal cross-correlation is an established method for estimating the
lag between paired physiological signals. The integer lag is the location of
the largest correlation coefficient. Fitting the correlation samples around
that maximum can estimate a sub-sample lag, but simple parabolic interpolation
can be biased and is therefore an estimator rather than an exact reconstruction.

## Decision

### Validated configuration and input

`TemporalSyncConfig` has private fields and a fallible constructor. It rejects:

- non-finite or non-positive frame spacing;
- a zero search range; and
- a non-finite minimum correlation outside `[0, 1]`.

Public getters expose the validated values. `TemporalSync::with_config`
continues to be infallible because it accepts only a constructed, valid
configuration.

Synchronization rejects unequal lengths, fewer than three samples, the first
non-finite sample in either input, and a signal whose full-sample variance is
not strictly positive. These failures use a public `TemporalSyncError` with
typed variants and offending values or indices.

### Correlation and lag convention

For each integer lag `k`, RITK computes the Pearson-normalized correlation over
the valid overlap:

```text
r(k) =
    Σᵢ (xᵢ - mean(xₖ)) (yᵢ₊ₖ - mean(yₖ))
    ------------------------------------------------
    sqrt(Σᵢ (xᵢ - mean(xₖ))² Σᵢ (yᵢ₊ₖ - mean(yₖ))²)
```

Positive `k` means that the moving signal `y` is delayed relative to reference
signal `x`; alignment samples `y` at coordinate `i + k`. A positive returned
shift therefore means “advance the moving signal by this amount.” Swapping the
signals negates the shift.

Peak selection uses one pass and constant search scratch. It keeps the best
integer lag, its correlation, and the adjacent correlations needed for
three-point parabolic refinement:

```text
δ = (r(k-1) - r(k+1)) / (2(r(k-1) - 2r(k) + r(k+1)))
```

The refinement is used only for a finite, non-flat local maximum and is clamped
to the neighboring interval. Boundary peaks remain integer-valued. Correlation
ties prefer the lag with the smallest absolute magnitude, then the smaller
signed lag, making the result deterministic.

The same lag iterator and normalized-correlation kernel back an explicitly
allocated `correlation_profile` diagnostic API. Profile allocation is absent
from `synchronize`.

### Result, acceptance, and residuals

The tuple and `TemporalQualityMetrics` contract are replaced by
`TemporalSyncResult`, which reports:

- shift in frames and seconds;
- peak normalized correlation;
- aligned overlap sample count;
- aligned RMS and maximum absolute residual in signal units; and
- `TemporalSyncStatus`, either accepted or below the configured correlation
  threshold.

The threshold classifies the returned estimate; it does not erase the measured
peak or turn a low-correlation observation into a numerical failure.

Aligned moving values use linear interpolation at `i + shift_frames`. Residual
metrics include only coordinates for which interpolation is defined and divide
RMS by that exact overlap count. They do not claim units of time.

`ImageRegistration::temporal_synchronization` and every export migrate to the
typed result. No tuple wrapper, old metric alias, or fallback remains.

## Consequences

- Non-finite, invalid, and unidentifiable inputs return typed errors instead of
  panicking or reporting false success.
- The configured threshold controls one explicit acceptance decision.
- Metric names and units match the computed quantities.
- The common synchronization path has constant search scratch; callers pay for
  a lag profile only when they request diagnostics.
- The return type, configuration construction, error type, and removal of
  `TemporalQualityMetrics` are breaking and require a major release when
  published.
- Parabolic sub-sample refinement remains a bounded estimator and its measured
  bias is exposed by analytical tests and the book example.

## Rejected alternatives

Filtering non-finite samples would silently change sample timing and overlap.
Treating flat signals as zero-lag success would continue to claim information
that the data do not contain. Returning an error below `min_correlation` would
discard a valid measured diagnostic. Retaining complete lag arrays in
`synchronize` would preserve unnecessary hot-path allocation. Labeling signal
residuals as seconds or scaling them by frame spacing would remain
dimensionally invalid.

FFT correlation is outside this slice: the current bounded search is direct,
and no profile shows transform setup as the binding cost. Unequal-rate
resampling and dynamic time warping are separate algorithms, not fallback
branches for this contract.

## Verification

Analytical tests cover zero, integer, and fractional delays; shift-sign symmetry;
positive affine-intensity invariance; valid-overlap counts; and independently
computed interpolated residuals. Negative tests assert every configuration and
input error variant. A table-driven differential test compares streaming peak
selection with the allocated diagnostic profile over seeded signals and search
ranges.

An unchanged Criterion workload measures the pre-change and post-change
implementation. The generated example asserts every displayed metric before
writing a figure with source signals, the lag profile, aligned traces, and
residuals.

## References

- Xiao, Ding, and Hu, “Time Synchronization of Multimodal Physiological
  Signals through Alignment of Common Signal Types and Its Technical
  Considerations in Digital Health,” *Journal of Imaging* 8(5), 120 (2022),
  Section 2.3, Equation 1, and Algorithm 1:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9145353/#sec2dot3-jimaging-08-00120>
- Céspedes, Huang, Ophir, and Spratt, “Methods for Estimation of Subsample Time
  Delays of Digitized Echo Signals,” *Ultrasonic Imaging* 17(2), 142–171
  (1995), abstract and curve-fitting comparison:
  <https://doi.org/10.1006/uimg.1995.1007>

## Revision history

- 2026-07-30: Initial accepted decision for SAFE-683-01.
