//! Spacing type for representing physical distances between pixels/voxels.
//!
//! `Spacing<D>` is a `#[repr(transparent)]` newtype over `Vector<D>` that
//! enforces domain separation: spacing values represent positive physical
//! distances (mm, µm, etc.), not general displacements. The zero-cost
//! wrapper preserves `Vector`'s `Copy`, `PartialEq`, arithmetic, and component
//! access through `Deref<Target = Vector<D>>`.
//!
//! # Invariant
//!
//! All components must be strictly positive (> 0). Spacing of zero is
//! physically meaningless (zero distance between adjacent voxels). This
//! invariant is enforced at construction time via `assert!` in `Spacing::new()`
//! and `Spacing::uniform()`; mutation through `DerefMut` is available for
//! compatibility with existing `spacing[i] = v` patterns, but callers must
//! not set non-positive values. Use `Spacing::new_unchecked()` only when the
//! caller can guarantee validity and the validation cost is unacceptable.

use super::Vector;
use burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::{AutodiffBackend, Backend};
use std::ops::{Deref, DerefMut};

/// Tolerance for determining whether voxel spacing is isotropic.
const ISOTROPY_TOLERANCE: f64 = 1e-9;

/// Error returned when `Spacing` construction receives non-positive components.
#[derive(Debug, Clone)]
pub struct InvalidSpacing {
    /// The index of the first offending component.
    pub index: usize,
    /// The invalid value that was rejected (≤ 0 or NaN).
    pub value: f64,
}

impl std::fmt::Display for InvalidSpacing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spacing component [{}] must be positive, got {}",
            self.index, self.value
        )
    }
}

impl std::error::Error for InvalidSpacing {}

/// Physical spacing between adjacent pixels/voxels along each axis.
///
/// A `#[repr(transparent)]` newtype over `Vector<D>` providing:
/// - **Domain separation**: `Spacing` ≠ `Vector` at the type level, preventing
///   accidental mixing of spacing values with displacement/velocity vectors.
/// - **Zero-cost abstraction**: identical memory layout and ABI to `Vector<D>`.
/// - **`Vector` API access**: `Deref` to `Vector<D>` provides indexing, arithmetic,
///   `to_array()`, `to_vec()`, etc.
/// - **Domain-specific methods**: `uniform()`, `is_uniform()`, `min_spacing()`,
///   `max_spacing()`.
///
/// # Construction
///
/// Use `Spacing::new([...])`, `Spacing::try_new([...])`, `Spacing::from_vector(v)`,
/// or `Spacing::uniform(s)`. `Spacing::new()` and `Spacing::uniform()` assert that
/// all components are strictly positive; `Spacing::try_new()` returns
/// `Result<Spacing<D>, InvalidSpacing>` for recoverable error handling.
/// `Spacing::new_unchecked()` skips validation for performance-critical paths
/// where the caller guarantees validity.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
pub struct Spacing<const D: usize>(Vector<D>);

impl<const D: usize> Spacing<D> {
    /// Create spacing from an array of component values.
    ///
    /// Panics if any component is ≤ 0 or NaN. For a non-panicking
    /// variant, see [`Spacing::try_new`]. For a variant that skips
    /// validation entirely, see [`Spacing::new_unchecked`].
    #[inline]
    pub fn new(components: [f64; D]) -> Self {
        Self::validate_positive(&components);
        Self(Vector::new(components))
    }

    /// Create spacing from an array of component values, returning
    /// `Err(InvalidSpacing)` if any component is ≤ 0 or NaN.
    #[inline]
    pub fn try_new(components: [f64; D]) -> Result<Self, InvalidSpacing> {
        for (i, &v) in components.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(InvalidSpacing { index: i, value: v });
            }
        }
        Ok(Self(Vector::new(components)))
    }

    /// Create spacing without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all components are strictly positive (> 0)
    /// and finite. Violating this invariant may cause incorrect behavior
    /// in downstream consumers that assume valid spacing (e.g., physical
    /// coordinate transforms, resampling, distance computations).
    #[inline]
    pub unsafe fn new_unchecked(components: [f64; D]) -> Self {
        #[cfg(debug_assertions)]
        for (i, &v) in components.iter().enumerate() {
            debug_assert!(
                v.is_finite() && v > 0.0,
                "Spacing::new_unchecked: component [{i}] must be positive and finite, got {v}"
            );
        }
        Self(Vector::new(components))
    }

    /// Create uniform spacing (same value for all dimensions).
    ///
    /// Asserts that `value > 0`. This is the common case for unit
    /// spacing (`Spacing::uniform(1.0)`).
    #[inline]
    pub fn uniform(value: f64) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "Spacing::uniform: value must be positive, got {value}"
        );
        Self(Vector::new(std::array::from_fn(|_| value)))
    }

    /// Create spacing from an existing `Vector<D>`.
    ///
    /// Asserts that all components of `v` are strictly positive, matching
    /// the invariant enforced by [`Spacing::new`].
    #[inline]
    pub fn from_vector(v: Vector<D>) -> Self {
        for i in 0..D {
            let val = v[i];
            assert!(
                val.is_finite() && val > 0.0,
                "Spacing::from_vector: component [{i}] must be positive, got {val}"
            );
        }
        Self(v)
    }

    /// Consume self and return the underlying `Vector<D>`.
    #[inline]
    pub fn into_vector(self) -> Vector<D> {
        self.0
    }

    /// Check if spacing is uniform (all components equal within tolerance).
    pub fn is_uniform(&self) -> bool {
        if D == 0 {
            return true;
        }
        let first = self[0];
        (1..D).all(|i| (self[i] - first).abs() < ISOTROPY_TOLERANCE)
    }

    /// Get the minimum spacing value.
    #[inline]
    pub fn min_spacing(&self) -> f64 {
        (0..D).map(|i| self[i]).fold(f64::INFINITY, |a, b| a.min(b))
    }

    /// Get the maximum spacing value.
    #[inline]
    pub fn max_spacing(&self) -> f64 {
        (0..D)
            .map(|i| self[i])
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Convert spacing to a fixed-size array (zero-allocation).
    #[inline]
    pub fn to_array(&self) -> [f64; D] {
        self.0.to_array()
    }

    /// Assert all components are strictly positive and finite.
    #[inline]
    fn validate_positive(components: &[f64; D]) {
        for (i, &v) in components.iter().enumerate() {
            assert!(
                v.is_finite() && v > 0.0,
                "Spacing::new: component [{i}] must be positive, got {v}"
            );
        }
    }
}

impl<const D: usize> Deref for Spacing<D> {
    type Target = Vector<D>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const D: usize> DerefMut for Spacing<D> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const D: usize> From<[f64; D]> for Spacing<D> {
    #[inline]
    fn from(arr: [f64; D]) -> Self {
        Self::new(arr)
    }
}

impl<const D: usize> From<Vector<D>> for Spacing<D> {
    #[inline]
    fn from(v: Vector<D>) -> Self {
        Self::from_vector(v)
    }
}

impl<const D: usize> From<Spacing<D>> for Vector<D> {
    #[inline]
    fn from(s: Spacing<D>) -> Self {
        s.0
    }
}

// ── Burn Module/Record impls (delegating to inner Vector) ─────────────────

impl<B: Backend, const D: usize> Record<B> for Spacing<D> {
    type Item<S: PrecisionSettings> = Spacing<D>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend, const D: usize> Module<B> for Spacing<D> {
    type Record = Self;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No tensors to visit
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        devices
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }
}

impl<B: AutodiffBackend, const D: usize> AutodiffModule<B> for Spacing<D> {
    type InnerModule = Spacing<D>;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl<const D: usize> ModuleDisplayDefault for Spacing<D> {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content.set_top_level_type(&format!("Spacing{}D", D)))
    }
}

impl<const D: usize> ModuleDisplay for Spacing<D> {}

#[cfg(test)]
#[path = "tests_spacing.rs"]
mod tests;
