//! Sliding-window percentile filter (see [`PercentileFilter`]).

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_morphology::{Offset3D, StructuringElement};
use ritk_tensor_ops::{extract_vec, rebuild};
use std::borrow::Cow;

/// Sliding-window percentile filter for 3-D volumes.
///
/// Replaces each voxel with the element at the requested percentile of its
/// structuring-element neighbourhood. Constructed from a percentile
/// `p ∈ [0, 100]` and a [`StructuringElement`] (typically `cube(r)`,
/// `cross(r)`, or `ball(r)`).
///
/// # Validation
/// The percentile must be in `[0, 100]`. A percentile outside this range
/// returns an `Err` from [`apply`](Self::apply).
#[derive(Clone, Debug)]
pub struct PercentileFilter {
    /// Percentile in `[0, 100]`.
    percentile: f32,
    /// Structuring element defining the neighbourhood.
    se: StructuringElement,
}

impl PercentileFilter {
    /// Construct a percentile filter with a cube SE of half-width `radius`.
    ///
    /// Equivalent to `scipy.ndimage.percentile_filter(input, percentile, size)`
    /// with `size = 2 * radius + 1` and a cubic footprint.
    #[inline]
    pub fn new(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::cube(radius))
    }

    /// Construct a percentile filter with a cube SE.
    #[inline]
    pub fn cube(percentile: f32, radius: usize) -> Self {
        Self::new(percentile, radius)
    }

    /// Construct a percentile filter with a cross SE.
    #[inline]
    pub fn cross(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::cross(radius))
    }

    /// Construct a percentile filter with a ball SE.
    #[inline]
    pub fn ball(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::ball(radius))
    }

    /// Construct a percentile filter from an explicit SE.
    ///
    /// The percentile must be in `[0, 100]`; an out-of-range value is
    /// rejected at [`apply`](Self::apply) time.
    #[inline]
    pub fn with_structuring_element(percentile: f32, se: StructuringElement) -> Self {
        Self { percentile, se }
    }

    /// Percentile in `[0, 100]`.
    #[inline]
    pub const fn percentile(&self) -> f32 {
        self.percentile
    }

    /// Structuring element reference.
    #[inline]
    pub fn structuring_element(&self) -> &StructuringElement {
        &self.se
    }

    /// Apply the percentile filter to a 3-D image.
    ///
    /// # Zero-copy fast path
    /// For a single-offset SE (only `radius = 0` produces this), the filter
    /// is the identity and the input is returned as `Cow::Borrowed(image)`.
    /// All other cases produce a freshly allocated `Image`.
    ///
    /// # Errors
    /// Returns `Err` if the percentile is not in `[0, 100]` or the tensor
    /// data cannot be cast to `f32`.
    pub fn apply<'a, B: Backend>(
        &self,
        image: &'a Image<B, 3>,
    ) -> anyhow::Result<Cow<'a, Image<B, 3>>> {
        if !(0.0..=100.0).contains(&self.percentile) || self.percentile.is_nan() {
            return Err(anyhow::anyhow!(
                "PercentileFilter: percentile must be in [0, 100], got {}",
                self.percentile
            ));
        }

        // Identity fast path: radius = 0 → SE = {(0,0,0)} → output = input.
        if self.se.is_empty() || self.se.len() == 1 {
            return Ok(Cow::Borrowed(image));
        }

        let (vals, shape) = extract_vec(image)?;
        let result = percentile_3d(&vals, shape, self.percentile, self.se.offsets());
        Ok(Cow::Owned(rebuild(result, shape, image)))
    }
}

/// Compute the percentile of every voxel's SE neighbourhood on a 3-D
/// `f32` volume stored in row-major `(Z, Y, X)` order.
///
/// # Boundary handling
/// Replicate (clamp) padding.
pub(crate) fn percentile_3d(
    data: &[f32],
    dims: [usize; 3],
    percentile: f32,
    se: &[Offset3D],
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = se.len();
    let rank_idx = ((percentile / 100.0) * ((n - 1) as f32)).floor() as usize;
    let rank_idx = rank_idx.min(n - 1);

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
                    scratch.select_nth_unstable_by(rank_idx, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *out_cell = scratch[rank_idx];
                }
            }
        },
    );

    output
}
