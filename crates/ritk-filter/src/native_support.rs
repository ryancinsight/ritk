//! Shared Coeus-`Image` boundary helper for this crate's pure flat-buffer
//! algorithm cores.
//!
//! Every Coeus-native filter wrapper in `ritk-filter` follows the same
//! extract → compute → reconstruct sequence: pull the contiguous voxel
//! buffer out of a [`ritk_image::native::Image`], hand it to an
//! already-substrate-agnostic pure function, and rebuild an `Image` from
//! the result with the source's spatial metadata preserved. `map_flat_image`
//! is the single generic entry point for that sequence (consolidation:
//! this crate's Euclidean-distance-transform wrapper duplicated the same
//! five lines before this module existed — the second occurrence is the
//! trigger to factor it out, not a third).

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

/// Apply a pure flat-buffer transform to a Coeus 3-D image, preserving its
/// shape and spatial metadata (origin, spacing, direction).
///
/// `f` receives the source's contiguous voxel data and `[nz, ny, nx]` shape
/// and returns a same-length output buffer. The length invariant is an
/// algorithm-specific contract enforced by each caller's differential
/// tests against its Burn-generic counterpart, not by this helper — this
/// function only owns the `Image` boundary, never the algorithm.
pub(crate) fn map_flat_image<B, F>(
    image: &Image<f32, B, 3>,
    backend: &B,
    f: F,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnOnce(&[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = image.shape();
    let vals = image.data_slice()?;
    let result = f(vals, dims);
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
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    F: FnOnce(&[f32], &[f32], [usize; 3]) -> Vec<f32>,
{
    let dims = primary.shape();
    anyhow::ensure!(
        dims == secondary.shape(),
        "native two-input filter: shape mismatch {:?} vs {:?}",
        dims,
        secondary.shape()
    );
    let a = primary.data_slice()?;
    let b = secondary.data_slice()?;
    let result = f(a, b, dims);
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

/// Test-only legacy Burn backend alias for filters whose type parameters are
/// independent from their Coeus-native execution path.
#[cfg(test)]
pub(crate) type LegacyBurnBackend = burn_ndarray::NdArray<f32>;

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

/// Shared differential-test harness for Coeus-native filter wrappers.
///
/// Runs a Burn-generic filter (`burn_apply`, on an `NdArray<f32>` image built
/// from `vals`/`dims`) and its Coeus-native counterpart (`coeus_apply`, on a
/// `SequentialBackend` image built from the same buffer) and asserts every
/// output voxel is bitwise-identical. Both sides of every wrapper in this
/// crate call the same substrate-agnostic core, so any divergence indicates
/// a boundary-marshaling bug, not an algorithmic one — no epsilon is
/// warranted (see numerical_discipline). Consolidated on the second
/// occurrence of this harness (the binary-morphology family) rather than
/// copied per filter.
#[cfg(test)]
pub(crate) fn assert_native_matches_burn<FB, FC>(
    vals: Vec<f32>,
    dims: [usize; 3],
    burn_apply: FB,
    coeus_apply: FC,
) where
    FB: FnOnce(
        &ritk_image::Image<burn_ndarray::NdArray<f32>, 3>,
    ) -> ritk_image::Image<burn_ndarray::NdArray<f32>, 3>,
    FC: FnOnce(
        &Image<f32, coeus_core::SequentialBackend, 3>,
        &coeus_core::SequentialBackend,
    ) -> Result<Image<f32, coeus_core::SequentialBackend, 3>>,
{
    use ritk_image::test_support as ts;
    use ritk_spatial::{Direction, Point, Spacing};

    let burn_image =
        ts::burn_compat::make_image::<burn_ndarray::NdArray<f32>, 3>(vals.clone(), dims);
    let burn_result = burn_apply(&burn_image);
    let burn_vals = burn_result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("burn result slice")
        .to_vec();

    let coeus_image = Image::from_flat_on(
        vals,
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &coeus_core::SequentialBackend,
    )
    .expect("coeus image construction");
    let coeus_result =
        coeus_apply(&coeus_image, &coeus_core::SequentialBackend).expect("coeus filter apply");
    let coeus_vals = coeus_result.data_slice().expect("coeus result slice");

    assert_eq!(
        coeus_vals.len(),
        burn_vals.len(),
        "coeus/burn output length mismatch"
    );
    for (i, (&c, &b)) in coeus_vals.iter().zip(burn_vals.iter()).enumerate() {
        assert_eq!(
            c, b,
            "coeus/burn divergence at flat index {i}: coeus={c}, burn={b}"
        );
    }
}

/// Two-input companion to [`assert_native_matches_burn`] for filters taking a
/// `(primary, secondary)` image pair (mask, geodesic reconstruction). Both sides
/// call the identical host core, so the outputs must be bitwise-identical.
#[cfg(test)]
pub(crate) fn assert_native_matches_burn_pair<FB, FC>(
    a_vals: Vec<f32>,
    b_vals: Vec<f32>,
    dims: [usize; 3],
    burn_apply: FB,
    coeus_apply: FC,
) where
    FB: FnOnce(
        &ritk_image::Image<burn_ndarray::NdArray<f32>, 3>,
        &ritk_image::Image<burn_ndarray::NdArray<f32>, 3>,
    ) -> ritk_image::Image<burn_ndarray::NdArray<f32>, 3>,
    FC: FnOnce(
        &Image<f32, coeus_core::SequentialBackend, 3>,
        &Image<f32, coeus_core::SequentialBackend, 3>,
        &coeus_core::SequentialBackend,
    ) -> Result<Image<f32, coeus_core::SequentialBackend, 3>>,
{
    use ritk_image::test_support as ts;

    let burn_a = ts::burn_compat::make_image::<burn_ndarray::NdArray<f32>, 3>(a_vals.clone(), dims);
    let burn_b = ts::burn_compat::make_image::<burn_ndarray::NdArray<f32>, 3>(b_vals.clone(), dims);
    let burn_result = burn_apply(&burn_a, &burn_b);
    let burn_vals = burn_result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("burn result slice")
        .to_vec();

    let coeus_a = make_native_image(a_vals, dims);
    let coeus_b = make_native_image(b_vals, dims);
    let coeus_result = coeus_apply(&coeus_a, &coeus_b, &coeus_core::SequentialBackend)
        .expect("coeus filter apply");
    let coeus_vals = coeus_result.data_slice().expect("coeus result slice");

    assert_eq!(
        coeus_vals.len(),
        burn_vals.len(),
        "coeus/burn output length mismatch"
    );
    for (i, (&c, &b)) in coeus_vals.iter().zip(burn_vals.iter()).enumerate() {
        assert_eq!(
            c, b,
            "coeus/burn pair divergence at flat index {i}: coeus={c}, burn={b}"
        );
    }
}
