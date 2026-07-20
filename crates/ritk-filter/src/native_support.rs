//! Shared Coeus-`Image` boundary helper for this crate's pure flat-buffer
//! algorithm cores.
//!
//! Every Coeus-native filter wrapper in `ritk-filter` follows the same
//! extract → compute → reconstruct sequence: pull the contiguous voxel
//! buffer out of a [`ritk_image::Image`], hand it to an
//! already-substrate-agnostic pure function, and rebuild an `Image` from
//! the result with the source's spatial metadata preserved. `map_flat_image`
//! is the single generic entry point for that sequence (consolidation:
//! this crate's Euclidean-distance-transform wrapper duplicated the same
//! five lines before this module existed — the second occurrence is the
//! trigger to factor it out, not a third).

use anyhow::Result;
use coeus_core::ComputeBackend;
use ritk_image::Image;

/// Apply a pure flat-buffer transform to a Coeus 3-D image, preserving its
/// shape and spatial metadata (origin, spacing, direction).
///
/// `f` receives the source's contiguous voxel data and `[nz, ny, nx]` shape
/// and returns a same-length output buffer. The length invariant is an
/// algorithm-specific contract enforced by each caller's differential
/// tests against its Coeus-generic counterpart, not by this helper — this
/// function only owns the `Image` boundary, never the algorithm.
pub(crate) fn map_flat_image<B, F>(
    image: &Image<f32, B, 3>,
    backend: &B,
    f: F,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    F: FnOnce(&[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = image.shape();
    let vals = image.try_data_vec_on(backend)?;
    let result = f(&vals, dims);
    Image::from_flat_on(
        result,
        dims,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

/// Apply a pure two-input flat-buffer transform to a pair of shape-matched
/// Coeus 3-D images, preserving the primary image's shape and spatial metadata.
///
/// `f` receives both source buffers and the shared `[nz, ny, nx]` shape and
/// returns a same-length output buffer built on `primary`'s metadata. The two
/// images must share a shape; a mismatch is a caller contract violation surfaced
/// as an error. This is the two-image companion to [`map_flat_image`], factored
/// out on its second occurrence (the mask and geodesic-reconstruction families).
pub(crate) fn map_flat_pair<B, F>(
    primary: &Image<f32, B, 3>,
    secondary: &Image<f32, B, 3>,
    backend: &B,
    f: F,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    F: FnOnce(&[f32], &[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = primary.shape();
    anyhow::ensure!(
        dims == secondary.shape(),
        "native two-input filter: shape mismatch {:?} vs {:?}",
        dims,
        secondary.shape()
    );
    let a = primary.try_data_vec_on(backend)?;
    let b = secondary.try_data_vec_on(backend)?;
    let result = f(&a, &b, dims);
    Image::from_flat_on(
        result,
        dims,
        *primary.origin(),
        *primary.spacing(),
        *primary.direction(),
        backend,
    )
}

/// Rebuild a Coeus-native image from a flat buffer, preserving the source image's
/// spatial metadata.
pub(crate) fn rebuild_image<B, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    src: &Image<f32, B, D>,
    backend: &B,
) -> Result<Image<f32, B, D>>
where
    B: ComputeBackend,
{
    ritk_tensor_ops::native::rebuild_image(vals, dims, src, backend)
}

/// Rebuild a Coeus-native image from a flat buffer and explicit spatial metadata.
pub(crate) fn rebuild_with_metadata<B, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    origin: ritk_spatial::Point<D>,
    spacing: ritk_spatial::Spacing<D>,
    direction: ritk_spatial::Direction<D>,
    _src: &Image<f32, B, D>,
    backend: &B,
) -> Result<Image<f32, B, D>>
where
    B: ComputeBackend,
{
    Image::from_flat_on(vals, dims, origin, spacing, direction, backend)
}

// ── Test infrastructure ───────────────────────────────────────────────────────

/// Construct a Coeus-native 3-D image from flat data with identity metadata.
///
/// Used by all native filter tests (ADR 0002 Batch #3 cutover).  The backend
/// is always `SequentialBackend` for unit tests; the single source of truth
/// for the pure `f32 → f32` algorithm is the flat buffer, not the substrate.
#[cfg(test)]
pub(crate) fn make_native_image(
    data: Vec<f32>,
    shape: [usize; 3],
) -> Image<f32, coeus_core::SequentialBackend, 3> {
    make_native_image_nd(data, shape)
}

/// Construct a Coeus-native image of any dimensionality with identity metadata.
#[cfg(test)]
pub(crate) fn make_native_image_nd<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
) -> Image<f32, coeus_core::SequentialBackend, D> {
    use ritk_spatial::{Direction, Point, Spacing};
    make_native_image_with_metadata(
        data,
        shape,
        Point::new([0.0; D]),
        Spacing::new([1.0; D]),
        Direction::identity(),
    )
}

/// Construct a Coeus-native image with explicit metadata.
#[cfg(test)]
pub(crate) fn make_native_image_with_metadata<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    origin: ritk_spatial::Point<D>,
    spacing: ritk_spatial::Spacing<D>,
    direction: ritk_spatial::Direction<D>,
) -> Image<f32, coeus_core::SequentialBackend, D> {
    Image::from_flat_on(
        data,
        shape,
        origin,
        spacing,
        direction,
        &coeus_core::SequentialBackend,
    )
    .expect("make_native_image_with_metadata: valid shape and data length")
}

/// Extract voxel data from a Coeus-native image as an owned `Vec<f32>`.
///
/// Panics if the image buffer is not C-contiguous (invariant of every image
/// that passes through the RITK filter pipeline).
#[cfg(test)]
pub(crate) fn native_vals(image: &Image<f32, coeus_core::SequentialBackend, 3>) -> Vec<f32> {
    native_vals_nd(image)
}

/// Dimensionality-generic companion to [`native_vals`].
#[cfg(test)]
pub(crate) fn native_vals_nd<const D: usize>(
    image: &Image<f32, coeus_core::SequentialBackend, D>,
) -> Vec<f32> {
    image
        .data_slice()
        .expect("native_vals: image must be contiguous")
        .to_vec()
}

/// Differentially compare the canonical image surface on Sequential with the
/// migration image surface on Moirai.
#[cfg(test)]
pub(crate) fn assert_coeus_matches_coeus<FB, FC>(
    vals: Vec<f32>,
    dims: [usize; 3],
    backend_a: FB,
    backend_b: FC,
) where
    FB: FnOnce(
        &ritk_image::Image<f32, coeus_core::SequentialBackend, 3>,
    ) -> ritk_image::Image<f32, coeus_core::SequentialBackend, 3>,
    FC: FnOnce(
        &Image<f32, coeus_core::MoiraiBackend, 3>,
        &coeus_core::MoiraiBackend,
    ) -> Result<Image<f32, coeus_core::MoiraiBackend, 3>>,
{
    let backend_a_input = ritk_image::Image::from_flat_on(
        vals.clone(),
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &coeus_core::SequentialBackend,
    )
    .expect("backend A fixture shape");
    let backend_b_provider = coeus_core::MoiraiBackend;
    let backend_b_input = Image::from_flat_on(
        vals,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &backend_b_provider,
    )
    .expect("backend B fixture shape");
    let backend_a_result = backend_a(&backend_a_input);
    let backend_b_result =
        backend_b(&backend_b_input, &backend_b_provider).expect("backend B filter apply");

    let vals_a = backend_a_result
        .data_slice()
        .expect("backend A result slice");
    let vals_b = backend_b_result
        .data_slice()
        .expect("backend B result slice");

    assert_eq!(vals_a.len(), vals_b.len(), "backend output length mismatch");
    for (i, (&a, &b)) in vals_a.iter().zip(vals_b.iter()).enumerate() {
        assert_eq!(a, b, "backend divergence at flat index {i}: A={a}, B={b}");
    }
}

/// Two-input companion to [`assert_coeus_matches_coeus`].
#[cfg(test)]
pub(crate) fn assert_coeus_matches_coeus_pair<FB, FC>(
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    dims: [usize; 3],
    backend_a: FB,
    backend_b: FC,
) where
    FB: FnOnce(
        &ritk_image::Image<f32, coeus_core::SequentialBackend, 3>,
        &ritk_image::Image<f32, coeus_core::SequentialBackend, 3>,
    ) -> ritk_image::Image<f32, coeus_core::SequentialBackend, 3>,
    FC: FnOnce(
        &Image<f32, coeus_core::MoiraiBackend, 3>,
        &Image<f32, coeus_core::MoiraiBackend, 3>,
        &coeus_core::MoiraiBackend,
    ) -> Result<Image<f32, coeus_core::MoiraiBackend, 3>>,
{
    let backend_a_lhs = ritk_image::Image::from_flat_on(
        lhs.clone(),
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &coeus_core::SequentialBackend,
    )
    .expect("backend A left fixture shape");
    let backend_a_rhs = ritk_image::Image::from_flat_on(
        rhs.clone(),
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &coeus_core::SequentialBackend,
    )
    .expect("backend A right fixture shape");
    let backend_b_provider = coeus_core::MoiraiBackend;
    let backend_b_lhs = Image::from_flat_on(
        lhs,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &backend_b_provider,
    )
    .expect("backend B left fixture shape");
    let backend_b_rhs = Image::from_flat_on(
        rhs,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
        &backend_b_provider,
    )
    .expect("backend B right fixture shape");
    let backend_a_result = backend_a(&backend_a_lhs, &backend_a_rhs);
    let backend_b_result = backend_b(&backend_b_lhs, &backend_b_rhs, &backend_b_provider)
        .expect("backend B filter apply");

    let vals_a = backend_a_result
        .data_slice()
        .expect("backend A result slice");
    let vals_b = backend_b_result
        .data_slice()
        .expect("backend B result slice");

    assert_eq!(vals_a.len(), vals_b.len(), "backend output length mismatch");
    for (i, (&a, &b)) in vals_a.iter().zip(vals_b.iter()).enumerate() {
        assert_eq!(a, b, "backend divergence at flat index {i}: A={a}, B={b}");
    }
}
