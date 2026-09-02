# ADR 0014: Fallible finite descriptive statistics

- Status: Accepted
- Date: 2026-07-30
- Board item: [SAFE-682-01](../../backlog.md#safe-682-01-major---make-descriptive-statistics-and-histograms-fallible-finite-and-teachable)

## Context

The public descriptive-statistics slice API indexed the first sample without
rejecting an empty input. Masked statistics asserted equal lengths and a
non-empty foreground. Histograms asserted a positive bin count and increasing
bounds. NaN samples bypassed range comparisons and were converted to bin zero,
while NaN in descriptive statistics entered arithmetic and was treated as
equal during percentile selection.

The normalization and Python layers depended on these infallible contracts.
Masked z-score normalization also substituted full-image statistics when a
mask selected no foreground, changing the caller's requested population
without reporting it.

The existing numerical contract uses an `f64` two-pass mean and variance, with
the NumPy divisor `N - ddof`. Quartiles are discrete floor-rank order
statistics at `N/4`, `N/2`, and `3N/4`; this differs from NumPy's default
linearly interpolated percentile method and is retained as an explicit RITK
convention.

## Decision

Descriptive statistics, histograms, and the normalization methods that depend
on them return `Result<_, StatisticsError>`. The error type distinguishes:

- empty input and empty masked foreground;
- non-finite image or mask samples, including their first invalid index;
- image/mask element-count mismatch;
- `ddof >= N`;
- zero histogram bins;
- non-finite or non-increasing histogram bounds; and
- histogram count-buffer allocation failure.

Validation completes before statistical arithmetic or histogram allocation.
The slice statistics path allocates one `O(N)` percentile workspace after
validation. The masked path collects foreground values once and reuses that
buffer for in-place percentile selection. Histogram bin coordinates execute
in `f64`, preventing finite extreme `f32` bounds from overflowing their span.
Count-buffer reservation is fallible. `Histogram::bin_width` returns `f64` so
the public accessor reports that same finite wide-arithmetic result.

The Coeus-native adapter delegates to the same slice functions. The Python
boundary maps `StatisticsError` to `ValueError`. An empty mask is an error in
masked z-score normalization; no full-image fallback remains. No infallible
wrapper is retained.

## Consequences

- Invalid external data no longer panics, silently enters bin zero, produces
  NaN output, or changes the requested masked population.
- The return-type changes are breaking and require a major release when
  published.
- Finite successful results preserve the established mean, variance, and
  floor-rank quartile formulas.
- Callers must handle the possibility that an image has no valid statistical
  population.

## Rejected alternatives

Filtering NaN and infinite samples would silently change the population.
Returning NaN for empty input or `ddof >= N` would defer a deterministic
contract error into downstream arithmetic. Clamping invalid histogram
parameters would change the requested bins. Retaining an infallible
normalization wrapper would preserve the same panic path under another API.

## Verification

Analytical tests cover known sequences, population and sample divisors,
permutation invariance, masked populations, histogram boundary inclusion, and
out-of-range exclusion. Negative tests cover every error variant relevant to
input data. Native and sequential normalization paths are differential-tested.
The generated book example cross-checks full and masked results against an
independent sorted reference, verifies histogram totals, and renders both
populations on one normalized axis.

## References

- NumPy `std`, including the `N - ddof` divisor:
  <https://numpy.org/doc/stable/reference/generated/numpy.std.html>
- NumPy `histogram`, including half-open bins and the inclusive final bin:
  <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>
- NumPy `percentile` method conventions:
  <https://numpy.org/doc/stable/reference/generated/numpy.percentile.html>
- Python `ValueError`:
  <https://docs.python.org/3/library/exceptions.html#ValueError>

## Revision history

- 2026-07-30: Initial accepted decision for SAFE-682-01.
