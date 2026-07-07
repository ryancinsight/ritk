//! Atlas-typed binary erosion filter for 3-D [`AtlasImage`].
//!
//! # Sub-batch #3.a (Atlas-meta Batch #3, ADR 0012)
//!
//! This module is the Atlas-typed sibling of [`crate::morphology::binary_erode`].
//! It defines [`AtlasBinaryErodeFilter`], which mirrors
//! [`crate::morphology::BinaryErodeFilter::apply`] semantics on the Atlas-typed
//! image carrier `AtlasImage<f32, MoiraiBackend, 3>` (i.e.
//! `ritk_image::native::Image<T, B, D>` from
//! `repos/ritk/crates/ritk-image/src/native.rs:18-25`).
//!
//! **Atomic-boundary invariant**: this module is strictly additive — the
//! legacy `BinaryErodeFilter` and its `apply<B: Backend>(&Image<B, 3>)`
//! signature remain untouched. Both surfaces coexist; consumer crates
//! migrate to `AtlasBinaryErodeFilter` per their own definition-of-ready.
//!
//! # Mathematical Specification
//!
//! Same [`crate::morphology::binary_erode::erode_binary_3d`] algorithm as
//! the legacy filter. We re-use that crate-private `pub(crate)` algorithm
//! via in-crate import. Output voxels are either the foreground value or
//! `0.0`. Boundary handling: out-of-bounds neighbours are treated as
//! background (ITK `BoundaryToForeground = false` default).
//!
//! # Why a sibling rather than a signature change
//!
//! Sub-batch #5 is the only commit authorised to delete or rename
//! `pub use …;` re-exports or narrow Burn-keyed public signatures
//! (per ADR 0012 §Decision §2). Migrating `BinaryErodeFilter::apply` to
//! accept `AtlasImage` would be a signature-narrowing change on a
//! Burn-keyed public surface — reserved for sub-batch #5. This sibling
//! migration is the additive per-crate shape until then.

use super::binary_erode::erode_binary_3d;
use super::types::ForegroundValue;
use coeus_core::storage::CpuAddressableStorage;
use coeus_core::ComputeBackend;
use ritk_image::AtlasImage;
use ritk_spatial::{Direction, Point, Spacing};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Atlas-typed binary erosion filter for 3-D [`AtlasImage`].
///
/// Mirrors [`crate::morphology::BinaryErodeFilter`] semantic shape, but
/// consumes (and produces) the Atlas-typed image carrier
/// `AtlasImage<T, B, D>` over `coeus_core::ComputeBackend`. The `D`
/// const-generic is fixed at `3` here, matching the legacy filter's
/// `apply<B>(&Image<B, 3>)` static-rank contract; per-crate migration to
/// lower-rank variants is tracked in subsequent sub-batches.
///
/// Construct via [`Self::new`] and custom-set foreground value via
/// [`Self::with_foreground`], matching the legacy API surfaces.
#[derive(Debug, Clone)]
pub struct AtlasBinaryErodeFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl AtlasBinaryErodeFilter {
    /// Create an Atlas-typed binary erosion filter with `radius` and
    /// default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Override the foreground value (mirrors [`crate::morphology::BinaryErodeFilter::with_foreground`]).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary erosion to a 3-D Atlas image.
    ///
    /// Returns a new Atlas image with identical shape and spatial
    /// metadata. Output voxels are `foreground_value` (foreground) or
    /// `0.0` (background).
    ///
    /// # Scalar-element constraint (sub-batch #3.a)
    ///
    /// The current implementation routes data through the
    /// [`erode_binary_3d`] algorithm (inherited from the legacy
    /// `binary_erode` module), which operates on `f32`. The Atlas-typed
    /// image carrier is fixed to `AtlasImage<f32, B, 3>` here, matching
    /// the legacy `BinaryErodeFilter::apply<B: Backend>(&Image<B, 3>)`
    /// shape where `Image<B, D>` is implicitly `f32` for medical-image
    /// carriers. Generalising the algorithm to `T: NumericElement` is
    /// deferred until `erode_binary_3d` itself is parameterised; per
    /// ADR 0012 §Decision §atomic-boundary discipline §2 (no public
    /// Burn-keyed surface mutation through sub-batch #4), the
    /// `pub(crate)` algorithm re-use preserves its current `f32`-only
    /// contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the host-data extraction step fails (e.g.
    /// non-contiguous AtlasImage layout). For the canonical
    /// `from_flat` construction path used by the per-crate sub-batch
    /// #3.a tests, layout is contiguous and this method never fails.
    pub fn apply<B>(&self, image: &AtlasImage<f32, B, 3>) -> anyhow::Result<AtlasImage<f32, B, 3>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let backend = B::default();
        let vals = image.data_vec_on(&backend);
        let dims = image.shape();
        let result = erode_binary_3d(&vals, dims, self.radius, self.foreground_value);

        let origin: Point<3> = *image.origin();
        let spacing: Spacing<3> = *image.spacing();
        let direction: Direction<3> = *image.direction();
        AtlasImage::from_flat_on(result, dims, origin, spacing, direction, &backend)
    }
}

impl Default for AtlasBinaryErodeFilter {
    fn default() -> Self {
        Self::new(1)
    }
}
