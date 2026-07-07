//! Atlas-keyed sister struct for binary erosion.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.c
//! (RITK-crate-migrate, per-crate atlas-typed migration queue #3.a..#3.g):
//! atlas-side parallel struct exposing host-slice binary erosion that bridges
//! from `AtlasImage<f32, MoiraiBackend, 3>` rasterized over
//! `coeus_tensor::Tensor`'s host accessor.
//!
//! Strictly additive on production surface per the sub-batch #3 invariant
//! (ADR 0012 §Decision §1): every public Burn-keyed symbol in
//! `super::BinaryErosion` (`pub struct BinaryErosion { radius }`,
//! `BinaryErosion::apply<B: Backend, const D: usize>(&self, &Image<B, D>) -> Image<B, D>`,
//! `impl MorphologicalOperation<B, D> for BinaryErosion`) is preserved verbatim.
//! This sister struct introduces a parallel host-slice forward path for
//! atlas-side callers that already operate on rasterized slices and do not
//! need the burn-tensor round-trip.

/// Atlas-side binary erosion filter.
///
/// Sister struct to `super::BinaryErosion`. Holds only the structuring-element
/// radius (the only state) and exposes a host-slice forward path so atlas-side
/// callers — primarily `AtlasImage<f32, MoiraiBackend, 3>` rasterizing over
/// `coeus_tensor::Tensor` — can exercise the same algorithm without crossing
/// the burn-tensor boundary.
///
/// All algorithmic semantics (structuring-element shape, D ∈ {1, 2, 3} support,
/// out-of-bounds-equals-foreground rule) are inherited from
/// `super::erode_nd`, the canonical CPU-side erosion kernel in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasBinaryErodeFilter {
    /// Half-width of the box structuring element in voxels.
    /// Radius 0 → structuring element = {p} → erosion is the identity.
    /// Radius 1 → 3^D neighbourhood.
    pub radius: usize,
}

impl AtlasBinaryErodeFilter {
    /// Create an `AtlasBinaryErodeFilter` with the given structuring-element radius.
    #[inline]
    pub const fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply erosion to a host-slice (row-major) binary mask.
    ///
    /// Argument contract:
    /// * `flat` — row-major flat `f32` buffer; `flat.len()` must equal
    ///   `shape.iter().product::<usize>()`. Caller guarantees this
    ///   (the atlas-side callers — `AtlasImage<f32, MoiraiBackend, 3>` over
    ///   `coeus_tensor::Tensor` — already know their raster shape).
    /// * `shape` — per-axis lengths; `shape.len()` selects dimensionality
    ///   (`shape.len() ∈ {1, 2, 3}` supported, other lengths panic via
    ///   `super::erode_nd`).
    /// * `radius` (struct field) — structuring-element half-width.
    ///
    /// Supports D ∈ {1, 2, 3} (delegated to `super::erode_nd → erode_line /
    /// erode_plane / erode_volume`). Out-of-bounds neighbours are treated as
    /// foreground (structuring element clipped to the image), matching ITK's
    /// `BinaryErodeImageFilter` default `BoundaryToForeground = true`.
    #[inline]
    pub fn apply(&self, flat: &[f32], shape: &[usize]) -> Vec<f32> {
        super::erode_nd(flat, shape, self.radius)
    }
}

impl Default for AtlasBinaryErodeFilter {
    #[inline]
    fn default() -> Self {
        Self::new(1)
    }
}
