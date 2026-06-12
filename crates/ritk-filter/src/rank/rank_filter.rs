//! Sliding-window rank filter (see [`RankFilter`]).

use ritk_tensor_ops::{extract_vec, rebuild};
use ritk_image::Image;
use ritk_morphology::{Offset3D, StructuringElement};
use burn::tensor::backend::Backend;
use std::borrow::Cow;

/// Sliding-window rank filter for 3-D volumes.
///
/// Replaces each voxel with the element at absolute position `rank` in the
/// sorted order of its structuring-element neighbourhood.
///
/// # Validation
/// `rank` must satisfy `rank < se.len()`. An out-of-range value returns
/// an `Err` from [`apply`](Self::apply).
#[derive(Clone, Debug)]
pub struct RankFilter {
    /// Absolute rank, 0-indexed, in `[0, se.len() - 1]`.
    rank: usize,
    /// Structuring element defining the neighbourhood.
    se: StructuringElement,
}

impl RankFilter {
    /// Construct a rank filter with a cube SE of half-width `radius`.
    #[inline]
    pub fn new(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::cube(radius))
    }

    /// Construct a rank filter with a cube SE.
    #[inline]
    pub fn cube(rank: usize, radius: usize) -> Self {
        Self::new(rank, radius)
    }

    /// Construct a rank filter with a cross SE.
    #[inline]
    pub fn cross(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::cross(radius))
    }

    /// Construct a rank filter with a ball SE.
    #[inline]
    pub fn ball(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::ball(radius))
    }

    /// Construct a rank filter from an explicit SE.
    #[inline]
    pub fn with_structuring_element(rank: usize, se: StructuringElement) -> Self {
        Self { rank, se }
    }

    /// Absolute rank, 0-indexed.
    #[inline]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Structuring element (clone of the SE; cheap).
    #[inline]
    pub fn structuring_element(&self) -> StructuringElement {
        self.se.clone()
    }

    /// Apply the rank filter to a 3-D image.
    ///
    /// # Zero-copy fast path
    /// For a single-offset SE (only `radius = 0` produces this), the filter
    /// is the identity and the input is returned as `Cow::Borrowed(image)`.
    ///
    /// # Errors
    /// Returns `Err` if `rank >= se.len()` or the tensor data cannot be cast
    /// to `f32`.
    pub fn apply<'a, B: Backend>(
        &self,
        image: &'a Image<B, 3>,
    ) -> anyhow::Result<Cow<'a, Image<B, 3>>> {
        if self.se.is_empty() {
            return Err(anyhow::anyhow!(
                "RankFilter: structuring element is empty (use a non-zero radius)"
            ));
        }
        if self.rank >= self.se.len() {
            return Err(anyhow::anyhow!(
                "RankFilter: rank {} out of range [0, {})",
                self.rank,
                self.se.len()
            ));
        }

        if self.se.len() == 1 {
            return Ok(Cow::Borrowed(image));
        }

        let (vals, shape) = extract_vec(image)?;
        let result = rank_select_3d(&vals, shape, self.rank, self.se.offsets());
        Ok(Cow::Owned(rebuild(result, shape, image)))
    }
}

/// Compute the element at absolute position `rank` in the sorted order of
/// every voxel's SE neighbourhood on a 3-D `f32` volume.
pub(crate) fn rank_select_3d(
    data: &[f32],
    dims: [usize; 3],
    rank: usize,
    se: &[Offset3D],
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = se.len();
    debug_assert!(rank < n, "rank {rank} out of range [0, {n})");

    let mut output = vec![0.0_f32; nz * ny * nx];
    let stride = ny * nx;

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        stride,
        |iz, out_slice| {
            let mut scratch: Vec<f32> = Vec::with_capacity(n);

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                for (ix, out_cell) in out_row.iter_mut().enumerate() {
                    scratch.clear();
                    for off in se {
                        let zz = (iz as i32 + off.iz()).clamp(0, nz as i32 - 1) as usize;
                        let yy = (iy as i32 + off.iy()).clamp(0, ny as i32 - 1) as usize;
                        let xx = (ix as i32 + off.ix()).clamp(0, nx as i32 - 1) as usize;
                        scratch.push(data[zz * stride + yy * nx + xx]);
                    }
                    scratch.select_nth_unstable_by(rank, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *out_cell = scratch[rank];
                }
            }
        },
    );

    output
}
