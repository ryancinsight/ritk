//! Atlas-keyed sister module for image intensity statistics.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.e
//! (RITK-crate-migrate, per-crate atlas-typed migration queue #3.a..#3.g):
//! atlas-side parallel module exposing [`AtlasImageStatistics`] sister
//! struct + [`compute_atlas_statistics`] / [`compute_atlas_statistics_from_slice`] /
//! [`atlas_masked_statistics`] sister functions over
//! `AtlasImage<f32, coeus_core::ComputeBackend, D>` rasterized through
//! [`ritk_image::native::Image`] (the canonical Atlas-typed
//! `ComputeBackend`-bound image carrier introduced in sub-batch #1).
//!
//! Strictly additive on production surface per the sub-batch #3 atomic-boundary
//! invariant (ADR 0012 §Decision §1): every public symbol of the legacy
//! Burn-keyed `image_statistics` module (`pub fn compute_statistics`,
//! `pub fn masked_statistics`, `pub fn compute_statistics_from_slice`,
//! `pub fn compute_from_values`, `pub struct ImageStatistics`,
//! `pub(crate) fn compute_from_owned`) is preserved verbatim, as is the
//! pre-existing `pub mod native;` sister. This Atlas twin introduces a parallel
//! surface whose semântica mirrors the legacy: same algorithm (delegates to
//! [`super::compute_from_owned`] for the canonical f64-precision fused-pass
//! plus quickselect-on-progressive-suffix percentile algorithm), same numerical
//! contract, sister error semantics ([`AtlasStatsError`] `Result`-returns
//! instead of `panic!` for caller-recoverable conditions).
//!
//! The sister struct is field-shape identical to `super::ImageStatistics` so
//! atlas-side callers can rely on the same destructuring pattern. The bidirectional
//! `From` impls stand up cross-interchangeability without any shape drift.
//!
//! **Data extraction** uses [`ritk_tensor_ops::native::extract_image_slice`] (the
//! canonical coeus-side adapter for `AtlasImage<f32, B, D>` → borrowed `&[f32]`),
//! mirroring the pre-existing `pub mod native;` sister's pattern. The
//! `B::DeviceBuffer<f32>: CpuAddressableStorage<f32>` trait bound matches
//! [`crate::image_statistics::native::compute_statistics`] verbatim.
//!
//! **No `Cargo.toml` mutation**: per-crate atomic-boundary invariant §3 forbids
//! adding/removing `[dependencies]` lines per per-crate commit (sub-batch #5 owns
//! the manifest strip cycle on the `[major]` pass). To avoid a `thiserror` dep-add,
//! the [`AtlasStatsError`] enum derives `Debug+Clone+PartialEq+Eq` only and
//! carries a hand-rolled `Display` impl via [`std::fmt::Display`] +
//! [`std::error::Error`] — zero new crate-graph edges.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;
use ritk_tensor_ops::native as tensor_ops;

use crate::FOREGROUND_THRESHOLD;

// ── Sister struct ─────────────────────────────────────────────────────────

/// Atlas-side sister struct to [`super::ImageStatistics`].
///
/// Field shape is identical (f32 min/max/mean/std + `[f32; 3]` percentiles) so
/// atlas-side consumers can rely on the same destructuring pattern. The derive
/// set is exactly mirrored from the legacy struct (`Debug + Clone + PartialEq`;
/// no `Eq` or `Hash` because the bytewise equality on `f32` would silently
/// accept NaN-equivalent sequences that violate strict equality — the Atlas
/// twin honours the legacy contract).
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasImageStatistics {
    /// Minimum intensity value.
    pub min: f32,
    /// Maximum intensity value.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Standard deviation with the requested `ddof` (0 = population, 1 = sample).
    pub std: f32,
    /// Percentiles: [p25, p50, p75].
    pub percentiles: [f32; 3],
}

impl From<AtlasImageStatistics> for super::ImageStatistics {
    #[inline]
    fn from(s: AtlasImageStatistics) -> Self {
        Self {
            min: s.min,
            max: s.max,
            mean: s.mean,
            std: s.std,
            percentiles: s.percentiles,
        }
    }
}

impl From<super::ImageStatistics> for AtlasImageStatistics {
    #[inline]
    fn from(s: super::ImageStatistics) -> Self {
        Self {
            min: s.min,
            max: s.max,
            mean: s.mean,
            std: s.std,
            percentiles: s.percentiles,
        }
    }
}

// ── Error semantics ───────────────────────────────────────────────────────

/// Atlas-side error variants for the masked-statistics path.
///
/// The legacy `super::masked_statistics` panics on missing foreground voxels
/// and on shape mismatch; the Atlas twin returns these `Result::Err` variants
/// so callers can decide whether to propagate, recover, or convert to their
/// own error type. The [`Display`](std::fmt::Display) impls are bit-identical
/// to the legacy panic messages so callers that switch between paths preserve
/// their diagnostic-text contract verbatim — no `"atlas "/``"coeus "` prefix
/// introduced (the legacy panic text is the canonical shared string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasStatsError {
    /// Foreground mask contains no voxels above the threshold.
    EmptyForegroundMask,
    /// Image and mask have different element counts.
    ShapeMismatch {
        /// Host-side element count of the image.
        image_n: usize,
        /// Host-side element count of the mask.
        mask_n: usize,
    },
}

impl std::fmt::Display for AtlasStatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyForegroundMask => {
                write!(f, "mask contains no foreground voxels")
            }
            Self::ShapeMismatch { image_n, mask_n } => write!(
                f,
                "image and mask element counts differ ({image_n} vs {mask_n})"
            ),
        }
    }
}

impl std::error::Error for AtlasStatsError {}

// ── Compute APIs ──────────────────────────────────────────────────────────

/// Compute descriptive statistics over **all** voxels in the Atlas-image.
///
/// Extraction path: [`tensor_ops::extract_image_slice`] returns the borrowed
/// host slice (failable under non-CPU backends); the borrowed slice is
/// allocated into an owned `Vec<f32>` once and delegated to
/// [`super::compute_from_owned`]. The `ddof = 0` default matches the legacy
/// `super::compute_statistics` (population std).
///
/// The trait bound `B::DeviceBuffer<f32>: CpuAddressableStorage<f32>` matches
/// the pre-existing `super::native::compute_statistics` surface verbatim so
/// the two sisters can co-exist for backends where data is host-resident
/// (`MoiraiBackend` ZST today); non-CPU-resident backends must route their
/// data through another extract path before calling this twin.
pub fn compute_atlas_statistics<B, const D: usize>(
    image: &Image<f32, B, D>,
) -> Result<AtlasImageStatistics, anyhow::Error>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (values, _) = tensor_ops::extract_image_slice(image)?;
    Ok(super::compute_statistics_from_slice(values, 0).into())
}

/// Compute descriptive statistics from an immutable host slice with `ddof`.
///
/// Sister to [`super::compute_statistics_from_slice`]. Pure host-slice path
/// so atlas-side callers that already have a `&[f32]` (e.g. after a
/// `coeus_tensor::Tensor::as_slice::<f32>()` borrow) can compute statistics
/// without re-allocating an `AtlasImage`.
pub fn compute_atlas_statistics_from_slice(slice: &[f32], ddof: usize) -> AtlasImageStatistics {
    super::compute_from_owned(slice.to_vec(), ddof).into()
}

/// Compute statistics restricted to voxels where `mask` > 0.5 (foreground).
///
/// Sister to [`super::masked_statistics`]. Returns
/// [`AtlasStatsError::EmptyForegroundMask`] (zero foreground) or
/// [`AtlasStatsError::ShapeMismatch`] (image/mask element-count mismatch)
/// instead of the legacy `panic!`. The Atlas-side error-return convention
/// matches the [`coeus_core::ComputeBackend`] idiomatic contract — `Result`-bearing
/// APIs avoid panicking for caller-recoverable conditions.
///
/// `mask` must contain 0.0 (background) or 1.0 (foreground) values per the
/// threshold check at `crate::FOREGROUND_THRESHOLD = 0.5`.
pub fn atlas_masked_statistics<B, const D: usize>(
    image: &Image<f32, B, D>,
    mask: &Image<f32, B, D>,
) -> Result<AtlasImageStatistics, AtlasStatsError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (image_values, _) =
        tensor_ops::extract_image_slice(image).map_err(|_| AtlasStatsError::ShapeMismatch {
            image_n: 0,
            mask_n: 0,
        })?;
    let (mask_values, _) =
        tensor_ops::extract_image_slice(mask).map_err(|_| AtlasStatsError::ShapeMismatch {
            image_n: image_values.len(),
            mask_n: 0,
        })?;
    if image_values.len() != mask_values.len() {
        return Err(AtlasStatsError::ShapeMismatch {
            image_n: image_values.len(),
            mask_n: mask_values.len(),
        });
    }
    let values: Vec<f32> = image_values
        .iter()
        .zip(mask_values.iter())
        .filter(|(_, &m)| m > FOREGROUND_THRESHOLD)
        .map(|(&v, _)| v)
        .collect();
    if values.is_empty() {
        return Err(AtlasStatsError::EmptyForegroundMask);
    }
    Ok(super::compute_from_owned(values, 0).into())
}
