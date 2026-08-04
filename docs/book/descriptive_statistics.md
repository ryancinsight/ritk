# Descriptive Statistics and Histograms

An image is both a spatial field and a population of intensity samples.
Descriptive statistics answer population questions before an algorithm uses
spatial relationships:

- What range of values exists?
- Where is the center of the distribution?
- How broad is the distribution?
- Does a foreground mask select a materially different population?

These questions are useful for quality control, choosing display windows,
normalization, and checking whether a mask contains the anatomy it claims to
contain. They do not explain where a value occurs; position extrema, label
statistics, and spatial filters answer different questions.

## The population contract

For finite samples `x₁, …, x_N`, RITK computes

```text
μ = (1 / N) Σᵢ₌₁ᴺ xᵢ
```

and

```text
σ = √(Σᵢ₌₁ᴺ (xᵢ − μ)² / (N − ddof))
```

Use `ddof = 0` for population standard deviation and `ddof = 1` for
sample standard deviation. This is the same divisor convention documented by
[NumPy `std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
RITK rejects `ddof >= N`; returning zero, infinity, or NaN would hide that no
valid divisor remains.

Mean and variance use two `f64` accumulation passes even though image samples
are `f32`. The wider accumulator prevents a long CT-scale sum from losing
individual voxel contributions once the running total's `f32` spacing exceeds
their magnitude. Results remain `f32` at the public image boundary.

## Quartile convention

RITK reports three discrete order statistics:

```text
Q₁ = x_(⌊N / 4⌋),  Q₂ = x_(⌊N / 2⌋),  Q₃ = x_(⌊3N / 4⌋)
```

Indices are zero-based after ordering the samples. These are observed sample
values, not interpolated estimates. NumPy exposes multiple
[percentile methods](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html);
its default linear method can therefore differ between adjacent samples.
RITK's floor-rank convention is stable and explicit.

The implementation does not fully sort the image. It copies a borrowed slice
once, then isolates the three ranks in place with progressively smaller
quickselect suffixes. The average complexity is `O(N)`, and the workspace is
one `f32` buffer of length `N`. Masked statistics collect foreground values
once and reuse that allocation as the percentile workspace.

## Full image and masked foreground

```rust,ignore
use ritk_statistics::{
    compute_statistics_from_slice,
    masked_statistics_from_slices,
};

let image = [12.0, 15.0, 82.0, 91.0, 154.0, 160.0];
let mask  = [ 0.0,  0.0,  1.0,  1.0,   1.0,   1.0];

let full = compute_statistics_from_slice(&image, 0)?;
let foreground = masked_statistics_from_slices(&image, &mask, 0)?;

assert!(foreground.mean > full.mean);
# Ok::<(), ritk_statistics::StatisticsError>(())
```

A mask value greater than `0.5` selects a foreground sample. Image and mask
buffers must have identical lengths. An empty foreground is an error; RITK
does not substitute full-image statistics because that changes the requested
population.

## Histograms

A histogram divides an explicit finite range `[min, max]` into equal-width
bins. Every bin except the last is half-open. The last includes `max`:

```text
B_k = [e_k, e_(k+1))       for k < K − 1
B_(K−1) = [e_(K−1), e_K]
```

This matches the edge convention documented by
[NumPy `histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html).
Finite values outside the range are excluded. NaN and infinite values are
rejected rather than discarded or assigned to bin zero.
`Histogram::bin_width()` returns `f64`, matching the wide bin-coordinate
arithmetic even when subtracting two finite `f32` bounds would overflow in
`f32`.

```rust,ignore
use ritk_statistics::histogram_from_slice;

let values = [0.0, 1.0, 2.0, 3.0, 4.0];
let histogram = histogram_from_slice(&values, 0.0, 4.0, 4)?;

assert_eq!(histogram.counts, vec![1, 1, 1, 2]);
assert_eq!(histogram.total(), values.len());
# Ok::<(), ritk_statistics::StatisticsError>(())
```

Histogram allocation is fallible. A hostile or accidental `bins` value that
cannot reserve its count buffer returns `HistogramAllocationFailed` instead of
relying on an unchecked allocation.

## Failure handling

`StatisticsError` distinguishes failures that require different caller
responses:

| Error | Meaning |
|---|---|
| `EmptyInput` | No population exists |
| `NonFiniteSample` | The first invalid image value and index |
| `NonFiniteMaskSample` | The first invalid mask value and index |
| `DegreesOfFreedomOutOfRange` | `N - ddof` is not positive |
| `ImageMaskLengthMismatch` | Image and mask cannot be paired |
| `EmptyForeground` | The mask selects no samples |
| `ZeroBins` | No histogram partition exists |
| `NonFiniteRange` | At least one histogram bound is NaN or infinite |
| `InvalidRange` | `min >= max` |
| `HistogramAllocationFailed` | The count buffer cannot be reserved |

Python bindings map these invalid values to `ValueError`, the Python category
for an argument of the correct type whose value violates the operation's
contract.

The [worked example](examples/descriptive_statistics.md) renders a source
field, foreground mask, normalized histogram comparison, and numeric
cross-checks from the same computed arrays.
