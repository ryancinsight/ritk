//! [`StructuringElement`] value type: a list of integer offsets plus the
//! half-width that produced them.
//!
//! # Mathematical Specification
//!
//! A structuring element (SE) is a finite subset of ℤ³ used as the footprint
//! of morphological operations (erosion, dilation, opening, closing, top-hat,
//! rank/percentile, etc.). An SE is represented by a list of integer offsets
//! `(Δz, Δy, Δx)` relative to the origin (0, 0, 0).
//!
//! # Origin Invariant
//!
//! `self.offsets` is guaranteed to contain `(0, 0, 0)` for any non-empty SE
//! produced by a shape constructor. The invariant is checked at construction
//! time via `debug_assert!`.
//!
//! # Zero-Copy
//!
//! The offset list is stored in a contiguous `Vec<Offset3D>`. The
//! [`StructuringElement::offsets`] accessor returns a `&[Offset3D]` slice that
//! is borrowed by callers (rank/percentile filters) without any copying.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic
//!   Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::offset::Offset3D;
use super::shape_markers::{Ball, Cross, Cube, SeShape};

/// A 3-D structuring element: a list of integer offsets plus the half-width
/// that produced them.
///
/// `StructuringElement` is the SSOT (single source of truth) for SE storage
/// in ritk-core. It is value-typed (`Clone + PartialEq + Eq + Default + Hash`),
/// allocation-light (a single `Vec<Offset3D>`), and referenceable via the
/// `offsets()` slice accessor for zero-copy reuse by hot loops.
///
/// # Origin Invariant
/// `self.offsets` is guaranteed to contain `(0, 0, 0)` for any non-empty SE
/// produced by a shape constructor. The invariant is checked at construction
/// time via `debug_assert!`.
///
/// # Example
/// ```ignore
/// use ritk_morphology::{StructuringElement, Cube, Cross, Ball};
///
/// // Cube 3×3×3 SE (r = 1, 27 voxels)
/// let cube = StructuringElement::cube(1);
/// assert_eq!(cube.len(), 27);
///
/// // Cross SE (r = 1, 7 voxels)
/// let cross = StructuringElement::cross(1);
/// assert_eq!(cross.len(), 7);
///
/// // Ball SE (r = 1, 7 voxels — same as cross for r=1)
/// let ball = StructuringElement::ball(1);
/// assert_eq!(ball.len(), 7);
///
/// // Ball SE (r = 2, 33 voxels)
/// let ball2 = StructuringElement::ball(2);
/// assert_eq!(ball2.len(), 33);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StructuringElement {
    /// Voxel offsets `(Δz, Δy, Δx)`. Always contains `(0, 0, 0)` when
    /// non-empty.
    offsets: Vec<Offset3D>,
    /// Half-width `r` that produced the offsets.
    radius: usize,
}

impl StructuringElement {
    /// Construct an SE from an explicit offset list and radius.
    ///
    /// The caller is responsible for the origin invariant; this constructor
    /// does not enforce it. Prefer the shape-specific constructors
    /// ([`cube`](Self::cube), [`cross`](Self::cross), [`ball`](Self::ball))
    /// unless a non-standard SE is genuinely required.
    #[inline]
    pub fn from_offsets(offsets: Vec<Offset3D>, radius: usize) -> Self {
        debug_assert!(
            radius == 0 || offsets.contains(&Offset3D::new(0, 0, 0)),
            "StructuringElement must contain the origin (0, 0, 0) when non-empty"
        );
        Self { offsets, radius }
    }

    /// Construct a cube SE of half-width `radius`.
    ///
    /// Cardinality: `(2r+1)³`. Delegated to [`Cube::offsets`].
    #[inline]
    pub fn cube(radius: usize) -> Self {
        Self {
            offsets: Cube::offsets(radius),
            radius,
        }
    }

    /// Construct a cross SE of half-width `radius`.
    ///
    /// Cardinality: `3(2r+1) - 2`. Delegated to [`Cross::offsets`].
    #[inline]
    pub fn cross(radius: usize) -> Self {
        Self {
            offsets: Cross::offsets(radius),
            radius,
        }
    }

    /// Construct a Euclidean ball SE of half-width `radius`.
    ///
    /// Cardinality: `#{x ∈ ℤ³ : ‖x‖₂ ≤ r}`. Delegated to [`Ball::offsets`].
    #[inline]
    pub fn ball(radius: usize) -> Self {
        Self {
            offsets: Ball::offsets(radius),
            radius,
        }
    }

    /// Half-width `r` that produced the SE.
    #[inline]
    pub const fn radius(&self) -> usize {
        self.radius
    }

    /// Borrow the SE offset list.
    ///
    /// Returns a `&[Offset3D]` — zero-copy. Callers (e.g. rank/percentile
    /// filters) may iterate the slice without allocating.
    #[inline]
    pub fn offsets(&self) -> &[Offset3D] {
        &self.offsets
    }

    /// Number of voxels in the SE.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the SE has no voxels (only true for `radius = 0` constructs).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Iterate over `(offset, index)` pairs.
    #[inline]
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, &Offset3D)> {
        self.offsets.iter().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `StructuringElement::iter_enumerated` yields every offset with a
    /// contiguous index `0..len`.
    #[test]
    fn iter_enumerated_indices_are_contiguous() {
        let se = StructuringElement::cube(1);
        let indices: Vec<usize> = se.iter_enumerated().map(|(i, _)| i).collect();
        assert_eq!(indices, (0..se.len()).collect::<Vec<_>>());
    }

    /// `StructuringElement::from_offsets` preserves the explicit list.
    #[test]
    fn from_offsets_preserves_list() {
        let offsets = vec![
            Offset3D::new(0, 0, 0),
            Offset3D::new(1, 0, 0),
            Offset3D::new(0, 1, 0),
        ];
        let se = StructuringElement::from_offsets(offsets.clone(), 1);
        assert_eq!(se.offsets(), &offsets[..]);
        assert_eq!(se.radius(), 1);
        assert_eq!(se.len(), 3);
    }

    /// `StructuringElement` is `Default` (empty SE with radius 0).
    #[test]
    fn default_is_empty() {
        let se = StructuringElement::default();
        assert!(se.is_empty());
        assert_eq!(se.radius(), 0);
        assert_eq!(se.len(), 0);
    }

    /// `cube`, `cross`, and `ball` constructors all produce a valid SE
    /// with non-zero radius and an offset list that includes the origin.
    #[test]
    fn shape_constructors_include_origin() {
        for r in 1..=3usize {
            let cube = StructuringElement::cube(r);
            let cross = StructuringElement::cross(r);
            let ball = StructuringElement::ball(r);
            assert_eq!(cube.radius(), r);
            assert_eq!(cross.radius(), r);
            assert_eq!(ball.radius(), r);
            assert!(!cube.is_empty());
            assert!(!cross.is_empty());
            assert!(!ball.is_empty());
            let origin = Offset3D::new(0, 0, 0);
            assert!(cube.offsets().contains(&origin));
            assert!(cross.offsets().contains(&origin));
            assert!(ball.offsets().contains(&origin));
        }
    }
}
