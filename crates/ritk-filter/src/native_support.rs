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
    use ritk_spatial::{Direction, Point, Spacing};
    Image::from_flat_on(
        data,
        shape,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &coeus_core::SequentialBackend,
    )
    .expect("make_native_image: valid shape and data length")
}

/// Extract voxel data from a Coeus-native image as an owned `Vec<f32>`.
///
/// Panics if the image buffer is not C-contiguous (invariant of every image
/// that passes through the RITK filter pipeline).
#[cfg(test)]
pub(crate) fn native_vals(image: &Image<f32, coeus_core::SequentialBackend, 3>) -> Vec<f32> {
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

    let burn_image = ts::make_image::<burn_ndarray::NdArray<f32>, 3>(vals.clone(), dims);
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
