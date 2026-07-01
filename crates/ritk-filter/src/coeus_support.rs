//! Shared Coeus-`Image` boundary helper for this crate's pure flat-buffer
//! algorithm cores.
//!
//! Every Coeus-native filter wrapper in `ritk-filter` follows the same
//! extract → compute → reconstruct sequence: pull the contiguous voxel
//! buffer out of a [`ritk_image::coeus::Image`], hand it to an
//! already-substrate-agnostic pure function, and rebuild an `Image` from
//! the result with the source's spatial metadata preserved. `map_flat_image`
//! is the single generic entry point for that sequence (consolidation:
//! this crate's Euclidean-distance-transform wrapper duplicated the same
//! five lines before this module existed — the second occurrence is the
//! trigger to factor it out, not a third).

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;

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
