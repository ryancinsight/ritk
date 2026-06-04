//! Zero-Sized Type markers and the sealed [`SeShape`] trait for canonical
//! 3-D structuring element shapes.
//!
//! # Type-Driven Dispatch
//!
//! The shape markers ([`Cube`], [`Cross`], [`Ball`]) are all Zero-Sized Types
//! (`size_of == 0`). They implement the sealed [`SeShape`] trait, which
//! enables compile-time shape selection and per-shape monomorphization.
//!
//! # Sealed Trait
//!
//! The [`SeShape`] trait is **sealed** — only [`Cube`], [`Cross`], and
//! [`Ball`] implement it. This protects the `(z, y, x)` row-major layout
//! invariant relied on by hot loops in downstream filters (rank/percentile,
//! grayscale morphology, etc.).
//!
//! # Const Cardinality
//!
//! [`cube_cardinality`], [`cross_cardinality`], and [`ball_cardinality_upper`]
//! are `const fn` evaluators used for stack-allocated offset arrays and
//! zero-overhead tests at compile time.

use super::offset::Offset3D;

// ── Sealed super-trait ────────────────────────────────────────────────────────

/// Sealed super-trait that bounds the [`SeShape`] implementations.
///
/// [`SeShape`] cannot be implemented outside this crate. This enforces:
/// 1. The `offsets` method must produce a `Vec<Offset3D>` containing exactly
///    `(0, 0, 0)` (the origin), matching the standard definition of a
///    structuring element.
/// 2. Implementations must be representable as Zero-Sized Types (no fields).
pub mod sealed {
    pub trait Sealed {}
}

// ── Const cardinality evaluators ──────────────────────────────────────────────

/// Number of voxels in a cubic SE of half-width `r`.
///
/// Cardinality: `(2r+1)³`. Evaluable in `const fn` contexts.
#[inline]
pub const fn cube_cardinality(r: usize) -> usize {
    let side = 2 * r + 1;
    side * side * side
}

/// Number of voxels in a cross SE of half-width `r`.
///
/// Cardinality: `3(2r+1) - 2` (three axis-aligned lines minus the doubly
/// counted origin). Evaluable in `const fn` contexts.
#[inline]
pub const fn cross_cardinality(r: usize) -> usize {
    3 * (2 * r + 1) - 2
}

/// Upper bound on the number of voxels in a ball SE of half-width `r`.
///
/// Equal to `cube_cardinality(r)` (the bounding cube). Exact ball count is
/// computed at runtime because it depends on integer-radius boundary
/// inclusion, but the upper bound is useful for stack-array pre-sizing.
#[inline]
pub const fn ball_cardinality_upper(r: usize) -> usize {
    cube_cardinality(r)
}

// ── SeShape trait ─────────────────────────────────────────────────────────────

/// Marker trait for the canonical 3-D structuring element shapes.
///
/// Each implementor is a Zero-Sized Type: `size_of::<Cube>() == 0`,
/// `size_of::<Cross>() == 0`, `size_of::<Ball>() == 0`. This means shape
/// selection is resolved entirely at compile time, with zero runtime cost.
///
/// The trait is **sealed** — only [`Cube`], [`Cross`], and [`Ball`] implement
/// it. This protects the `(z, y, x)` row-major layout invariant relied on by
/// hot loops in downstream filters.
///
/// # Monomorphization
///
/// Each shape produces a separate monomorphized function at every call site,
/// enabling LLVM to constant-fold the offset-generation loops when the radius
/// is also a compile-time constant.
pub trait SeShape: Copy + Clone + Default + Send + Sync + sealed::Sealed {
    /// Human-readable shape name (e.g. `"cube"`, `"cross"`, `"ball"`).
    const NAME: &'static str;

    /// Generate the SE offsets for a given half-width `radius`.
    ///
    /// Always includes the origin `(0, 0, 0)`. Offsets are emitted in
    /// deterministic row-major order: `iz` varies slowest, `ix` fastest.
    fn offsets(radius: usize) -> Vec<Offset3D>;
}

// ── Cube ──────────────────────────────────────────────────────────────────────

/// Cube structuring element: `(2r+1)³` axis-aligned cubic neighbourhood.
///
/// Includes all offsets `(Δz, Δy, Δx)` with `|Δi| ≤ r` for `i ∈ {z, y, x}`.
///
/// # Cardinality
/// `(2r+1)³` — see [`cube_cardinality`].
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Cube;

impl sealed::Sealed for Cube {}

impl SeShape for Cube {
    const NAME: &'static str = "cube";

    #[inline]
    fn offsets(radius: usize) -> Vec<Offset3D> {
        let r = radius as i32;
        let cap = cube_cardinality(radius);
        let mut out = Vec::with_capacity(cap);
        for iz in -r..=r {
            for iy in -r..=r {
                for ix in -r..=r {
                    out.push(Offset3D::new(iz, iy, ix));
                }
            }
        }
        debug_assert_eq!(out.len(), cap, "cube cardinality must match (2r+1)³");
        out
    }
}

// ── Cross ─────────────────────────────────────────────────────────────────────

/// Cross structuring element: union of three axis-aligned lines of length
/// `2r+1` centred on the origin.
///
/// Includes offsets `(Δz, Δy, Δx)` with at most one non-zero component whose
/// absolute value is at most `r`. The origin `(0, 0, 0)` is included exactly
/// once.
///
/// # Cardinality
/// `3(2r+1) - 2` — see [`cross_cardinality`].
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Cross;

impl sealed::Sealed for Cross {}

impl SeShape for Cross {
    const NAME: &'static str = "cross";

    #[inline]
    fn offsets(radius: usize) -> Vec<Offset3D> {
        let r = radius as i32;
        let cap = cross_cardinality(radius);
        let mut out = Vec::with_capacity(cap);
        for i in -r..=r {
            out.push(Offset3D::new(i, 0, 0));
        }
        for i in -r..=r {
            if i == 0 {
                continue;
            }
            out.push(Offset3D::new(0, i, 0));
        }
        for i in -r..=r {
            if i == 0 {
                continue;
            }
            out.push(Offset3D::new(0, 0, i));
        }
        debug_assert_eq!(out.len(), cap, "cross cardinality must match 3(2r+1)-2");
        out
    }
}

// ── Ball ──────────────────────────────────────────────────────────────────────

/// Ball structuring element: Euclidean ball of radius `r` in voxel units.
///
/// Includes offsets `(Δz, Δy, Δx)` with `Δz² + Δy² + Δx² ≤ r²`. Equivalently
/// the L²-closed ball intersected with ℤ³. The origin `(0, 0, 0)` is
/// always included.
///
/// # Cardinality
/// `#{x ∈ ℤ³ : ‖x‖₂ ≤ r}` — computed at runtime. For a rough upper bound see
/// [`ball_cardinality_upper`].
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Ball;

impl sealed::Sealed for Ball {}

impl SeShape for Ball {
    const NAME: &'static str = "ball";

    #[inline]
    fn offsets(radius: usize) -> Vec<Offset3D> {
        let r = radius as i32;
        let r_sq = r * r;
        let upper = ball_cardinality_upper(radius);
        let mut out = Vec::with_capacity(upper);
        for iz in -r..=r {
            for iy in -r..=r {
                for ix in -r..=r {
                    let d_sq = iz * iz + iy * iy + ix * ix;
                    if d_sq <= r_sq {
                        out.push(Offset3D::new(iz, iy, ix));
                    }
                }
            }
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Cube cardinality formula: `(2r+1)³` for r ∈ {0, 1, 2, 3, 5}.
    #[test]
    fn cube_cardinality_matches_formula() {
        for r in 0..=5usize {
            let expected = (2 * r + 1).pow(3);
            assert_eq!(
                Cube::offsets(r).len(),
                expected,
                "cube(r={r}) must have (2r+1)³ = {expected} offsets"
            );
            assert_eq!(
                cube_cardinality(r),
                expected,
                "cube_cardinality(r={r}) must equal (2r+1)³"
            );
        }
    }

    /// Cross cardinality formula: `3(2r+1) - 2`.
    #[test]
    fn cross_cardinality_matches_formula() {
        for r in 0..=5usize {
            let expected = 3 * (2 * r + 1) - 2;
            assert_eq!(
                Cross::offsets(r).len(),
                expected,
                "cross(r={r}) must have 3(2r+1)-2 = {expected} offsets"
            );
            assert_eq!(
                cross_cardinality(r),
                expected,
                "cross_cardinality(r={r}) must equal 3(2r+1)-2"
            );
        }
    }

    /// Ball cardinality for small radii matches known values.
    ///
    /// # Derivation
    /// `ball(r)` = `#{x ∈ ℤ³ : ‖x‖₂² ≤ r²}`. The count by squared distance:
    /// - d²=0: 1 (origin)
    /// - d²=1: 6 (face-centre points (±1,0,0), (0,±1,0), (0,0,±1))
    /// - d²=2: 12 (edge-midpoints (±1,±1,0) etc, 3 axes × 4 sign combos)
    /// - d²=3: 8 (corner points (±1,±1,±1))
    /// - d²=4: 6 (face-centre points (±2,0,0) etc)
    ///
    /// Total for r=1: 1+6 = 7. Total for r=2: 1+6+12+8+6 = 33.
    #[test]
    fn ball_cardinality_matches_known_values() {
        assert_eq!(Ball::offsets(0).len(), 1, "ball(0) = origin");
        assert_eq!(Ball::offsets(1).len(), 7, "ball(1) = origin + 6 face centres");
        assert_eq!(
            Ball::offsets(2).len(),
            33,
            "ball(2) = origin + 6 d²=1 + 12 d²=2 + 8 d²=3 + 6 d²=4"
        );
    }

    /// All three shape markers are Zero-Sized Types.
    #[test]
    fn shape_markers_are_zsts() {
        assert_eq!(std::mem::size_of::<Cube>(), 0, "Cube must be a ZST");
        assert_eq!(std::mem::size_of::<Cross>(), 0, "Cross must be a ZST");
        assert_eq!(std::mem::size_of::<Ball>(), 0, "Ball must be a ZST");
    }

    /// The origin `(0, 0, 0)` is included in every non-empty shape.
    #[test]
    fn origin_is_included_in_every_shape() {
        let origin = Offset3D::new(0, 0, 0);
        for r in 0..=3usize {
            assert!(Cube::offsets(r).contains(&origin), "cube r={r} must contain origin");
            assert!(Cross::offsets(r).contains(&origin), "cross r={r} must contain origin");
            assert!(Ball::offsets(r).contains(&origin), "ball r={r} must contain origin");
        }
    }

    /// Offsets are deterministic across multiple invocations.
    #[test]
    fn offsets_are_deterministic() {
        let a = Cube::offsets(2);
        let b = Cube::offsets(2);
        assert_eq!(a, b, "Cube::offsets must be deterministic");

        let a = Cross::offsets(2);
        let b = Cross::offsets(2);
        assert_eq!(a, b, "Cross::offsets must be deterministic");

        let a = Ball::offsets(2);
        let b = Ball::offsets(2);
        assert_eq!(a, b, "Ball::offsets must be deterministic");
    }
}
