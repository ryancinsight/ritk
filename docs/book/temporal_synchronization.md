# Temporal Signal Synchronization

Spatial registration answers “where is this anatomy?” Temporal
synchronization answers “when did this event occur?” A dynamic MR sequence,
PET time-activity curve, physiological trace, or gated reconstruction can be
spatially aligned while its samples still refer to different moments.

RITK estimates one constant delay between two equally sampled finite signals.
It does not resample unequal acquisition rates, model a delay that changes over
time, or perform dynamic time warping. Those are different contracts.

## Lag convention

Let `xᵢ` be the reference signal and `yᵢ` the moving signal. At integer
lag `k`, RITK evaluates only the index pairs that exist in both arrays:

```text
r(k) = Σᵢ (xᵢ − x̄_k)(yᵢ₊ₖ − ȳ_k)
       / √(Σᵢ (xᵢ − x̄_k)² · Σᵢ (yᵢ₊ₖ − ȳ_k)²)
```

The overlap-specific means prevent a shorter edge overlap from being compared
against full-signal means. Pearson normalization also makes the estimated lag
invariant to a positive scale and offset applied to either signal.

A **positive** `k` means the moving signal is delayed. To align it, sample the
moving signal at `i + k`. The reported shift in seconds is

```text
Δt = k · T_frame
```

This sign convention is visible in the [worked
example](examples/temporal_synchronization.md): the unaligned orange curve
lags the reference, and the aligned curve is evaluated at
`reference_index + shift_frames`.

## From integer peak to fractional estimate

The integer lag with maximum normalized correlation is the discrete estimate.
When correlations exist on both sides of an interior maximum, RITK fits a
three-point parabola:

```text
δ = [r(k − 1) − r(k + 1)] / [2(r(k − 1) − 2r(k) + r(k + 1))]
k̂ = k + δ
```

The offset is accepted only for a finite concave peak and is bounded to the
neighboring interval. A peak at the configured search boundary remains
integer-valued. Parabolic refinement is an estimator; digitized correlation
curves can bias it. Céspedes et al. compare parabolic and cosine fits and
describe this limitation in [*Methods for Estimation of Subsample Time Delays
of Digitized Echo Signals*](https://doi.org/10.1006/uimg.1995.1007).

Correlation ties are deterministic: RITK chooses the smallest absolute lag,
then the smaller signed lag. This prevents scheduling or iterator order from
changing a result.

## Validated configuration and typed failures

```rust,ignore
use leto::Array1;
use ritk_registration::{
    TemporalSync, TemporalSyncConfig, TemporalSyncStatus,
};

let reference = Array1::from_vec(
    [8],
    vec![0.0, 0.2, 0.9, 0.4, -0.3, -0.8, -0.2, 0.3],
)?;
let moving = Array1::from_vec(
    [8],
    vec![0.0, 0.0, 0.2, 0.9, 0.4, -0.3, -0.8, -0.2],
)?;

let config = TemporalSyncConfig::try_new(
    0.04, // seconds per sample
    3,    // search ±3 frames
    0.8,  // acceptance threshold
)?;
let result = TemporalSync::with_config(config)
    .synchronize(&reference, &moving)?;

assert!(result.shift_frames() > 0.0);
assert_eq!(result.status(), TemporalSyncStatus::Accepted);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`TemporalSyncConfig::try_new` rejects non-finite or non-positive frame
spacing, a zero search range, and a minimum correlation outside `[0, 1]`.
Synchronization then rejects:

- unequal signal lengths;
- fewer than three samples;
- the first NaN or infinite sample, with signal identity and index; and
- a zero-variance signal whose lag is not identifiable.

Two identical constant signals are not “perfectly synchronized.” They contain
no timing feature, so every lag explains them equally well. RITK reports
`TemporalSyncError::UnidentifiableSignal` instead of inventing a zero shift.

## Acceptance is not numerical success

`minimum_correlation` classifies a measured estimate:

- `TemporalSyncStatus::Accepted`, or
- `TemporalSyncStatus::BelowMinimumCorrelation`.

A below-threshold result still exposes its shift and correlation. This lets a
quality-control pipeline display or record the observation without confusing
weak evidence with a numerical failure.

The correlation coefficient can be negative. The acceptance threshold is
non-negative because a negatively correlated peak is not evidence that the
signals share the same polarity and timing.

## Residual metrics and units

After estimating `k̂`, RITK linearly interpolates `y` at `i + k̂`. Only
coordinates inside the moving signal contribute:

```text
RMS = √((1 / N_overlap) · Σᵢ∈overlap [xᵢ − ỹ(i + k̂)]²)
```

`residual_rms` and `residual_max_abs` are in **signal amplitude units**, not
seconds. `shift_seconds` is the timing quantity. `overlap_samples` states the
exact RMS denominator, so a large shift cannot make the error look smaller by
dividing by samples that were never compared.

## Search allocation and diagnostic profiles

`synchronize` scans lags once and retains only the current best peak and its
neighbors. Search scratch is constant-size; it does not allocate full lag and
correlation arrays. Call `correlation_profile` only when a graph or diagnostic
record needs every lag:

```rust,ignore
let profile = synchronizer.correlation_profile(&reference, &moving)?;
for sample in &profile {
    if let Some(correlation) = sample.correlation() {
        println!("lag={} r={correlation}", sample.lag_frames());
    }
}
# Ok::<(), ritk_registration::TemporalSyncError>(())
```

A profile sample is `None` when that particular overlap is locally constant.
The normal peak selector ignores it. Full-signal zero variance remains a typed
input error.

The direct method performs `O(N · S)` work for `N` samples and search radius
`S`. It is appropriate when the physically plausible delay window is small.
The committed `temporal_sync` benchmark covers 4,096 samples and a ±64-frame
search without changing the workload between implementations.

## Method reference

Xiao, Ding, and Hu describe lagged cross-correlation and maximum-lag selection
for multimodal physiological signals in [Section 2.3, Equation 1, and
Algorithm 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9145353/#sec2dot3-jimaging-08-00120).
RITK pins its overlap, normalization, tie, sign, interpolation, and failure
conventions above because a method name alone does not define those choices.
