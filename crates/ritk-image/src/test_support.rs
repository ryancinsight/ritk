//! Test-only constructors for [`Image`].
//!
//! Consolidates the ~70 ad-hoc `make_image_*`/`make_mask_*` test helpers
//! scattered across every crate's `tests_*.rs` files behind a single
//! family of const-generic entry points. One implementation serves
//! `D ∈ {1, 2, 3, 4}` and the optional backend `B`.
//!
//! # Variants
//!
//! - [`make_image`] — minimum surface: zero origin, unit spacing, identity
//!   direction. Right for ~95% of test cases that don't depend on spatial
//!   metadata.
//! - [`make_image_with`] — full surface: every spatial field overridable.
//!   Use when the test depends on anisotropy, off-origin indexing, or a
//!   non-identity direction matrix.
//! - [`make_image_with_spacing`] — convenience for the common case of
//!   non-unit spacing (kept here to avoid per-file re-implementations).
//! - [`fill_image`] — pure-fill constructor: produces an image with the
//!   given value at every voxel, sized `[d0, …, d{D-1}]`.
//!
//! All helpers return `Image<f32, B, D>` and so require `B: Backend +
//! Default`. Consumer tests fix `B` at the binding site:
//!
//! ```ignore
//! use coeus_core::SequentialBackend;
//! use ritk_image::test_support::make_image;
//! type TestBackend = SequentialBackend;
//! let img: Image<TestBackend, 3> = make_image(vec![0.0; 24], [2, 3, 4]);
//! ```
//!
//! Module is feature-gated because crate-local `#[cfg(test)]` does NOT
//! propagate to dependent crates' test binaries. The declaration in
//! `lib.rs` uses `#[cfg(any(test, feature = "test-helpers"))]` so a
//! downstream test crate can opt in via the `test-helpers` feature while
//! release artifacts keep these helpers out of the normal build.
//!
//! [`Image`]: crate::Image

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

use crate::types::Image;

fn make_tensor<T, B, const D: usize>(data: Vec<T>, dims: [usize; D]) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    Tensor::<T, B>::from_slice_on(dims, &data, &backend)
}

/// Build an [`Image<T, B, D>`] from raw voxel data with default spatial metadata.
///
/// Defaults: zero origin, unit spacing (1.0 per axis), identity direction.
///
/// # Type Parameters
/// * `T` — scalar element type (e.g. `f32`).
/// * `B` — Coeus compute backend (e.g. `SequentialBackend`).
/// * `D` — image rank, must satisfy `D > 0`.
///
/// # Arguments
/// * `data` — flat voxel values in row-major order.
/// * `dims` — per-axis extent; `data.len()` MUST equal `dims.iter().product()`.
///
/// # Panics
/// Panics if `data.len() != dims.iter().product()`.
pub fn make_image<T, B, const D: usize>(data: Vec<T>, dims: [usize; D]) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    make_image_with(data, dims, None, None, None)
}

/// Build a 1D [`Image<T, B, 1>`] from raw voxel data with default spatial metadata.
pub fn make_image_1d<T, B>(data: Vec<T>) -> Image<T, B, 1>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let n = data.len();
    make_image(data, [n])
}

/// Build an [`Image<T, B, D>`] from raw voxel data with overridable spatial
/// metadata. `None` fields use the same defaults as [`make_image`].
pub fn make_image_with<T, B, const D: usize>(
    data: Vec<T>,
    dims: [usize; D],
    origin: Option<Point<D>>,
    spacing: Option<Spacing<D>>,
    direction: Option<Direction<D>>,
) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let origin = origin.unwrap_or_else(|| Point::new([0.0_f64; D]));
    let spacing = spacing.unwrap_or_else(|| Spacing::new([1.0_f64; D]));
    let direction = direction.unwrap_or_else(Direction::identity);
    Image::new(make_tensor::<T, B, D>(data, dims), origin, spacing, direction)
}

/// Build an [`Image<T, B, D>`] from raw voxel data with custom per-axis spacing
/// and otherwise default spatial metadata.
pub fn make_image_with_spacing<T, B, const D: usize>(
    data: Vec<T>,
    dims: [usize; D],
    spacing: [f64; D],
) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    make_image_with(data, dims, None, Some(Spacing::new(spacing)), None)
}

/// Build an [`Image<T, B, D>`] filled with a single value everywhere.
pub fn fill_image<T, B, const D: usize>(dims: [usize; D], value: T) -> Image<T, B, D>
where
    T: Scalar + Clone,
    B: ComputeBackend + Default,
{
    let n: usize = dims.iter().product();
    make_image(vec![value; n], dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    #[test]
    fn make_image_1d_default_origin_spacing_direction() {
        let img: Image<f32, B, 1> = make_image(vec![1.0_f32, 2.0, 3.0], [3]);
        assert_eq!(img.shape(), [3]);
        assert_eq!(img.origin()[0], 0.0);
        assert_eq!(img.spacing()[0], 1.0);
    }

    #[test]
    fn make_image_3d_default_origin_spacing_direction() {
        let img: Image<f32, B, 3> = make_image(vec![0.0_f32; 24], [2, 3, 4]);
        assert_eq!(img.shape(), [2, 3, 4]);
        assert_eq!(img.origin()[0], 0.0);
        assert_eq!(img.spacing()[2], 1.0);
    }

    #[test]
    fn make_image_with_overrides_are_applied() {
        let img: Image<f32, B, 2> = make_image_with(
            vec![1.0_f32; 6],
            [2, 3],
            Some(ritk_spatial::Point::new([1.0, 2.0])),
            Some(ritk_spatial::Spacing::new([0.5, 0.25])),
            None,
        );
        assert_eq!(img.origin()[0], 1.0);
        assert_eq!(img.spacing()[1], 0.25);
    }

    #[test]
    fn make_image_with_spacing_applies_all_axes() {
        let img: Image<f32, B, 3> =
            make_image_with_spacing(vec![0.0_f32; 8], [2, 2, 2], [0.5, 1.5, 2.5]);
        assert_eq!(img.spacing()[0], 0.5);
        assert_eq!(img.spacing()[1], 1.5);
        assert_eq!(img.spacing()[2], 2.5);
        assert_eq!(img.origin()[0], 0.0);
    }

    #[test]
    fn fill_image_constant_value() {
        let img: Image<f32, B, 2> = fill_image([3, 4], 7.0);
        assert_eq!(img.shape(), [3, 4]);
        let v = img.data().as_slice().to_vec();
        assert!(v.iter().all(|&x| x == 7.0));
    }
}
