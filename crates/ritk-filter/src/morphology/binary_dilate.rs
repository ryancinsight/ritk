//! Binary dilation filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary dilation with a flat cubic structuring element B of half-width r:
//!
//!   (D_B f)(x) = fg  iff  ∃ b ∈ B: f(x − b) = fg
//!             = bg  otherwise
//!
//! Equivalently: a voxel x is foreground in the output iff any voxel in its
//! `(2r+1)³` cubic neighbourhood is foreground in the input.
//!
//! # Boundary Handling
//!
//! Out-of-bounds positions are treated as background — they do not contribute
//! foreground.  This is equivalent to `itk::BinaryDilateImageFilter` with
//! `BoundaryToForeground = false` (the ITK default).
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryDilateImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetBackgroundValue(0.0)`
//! - `SetBoundaryToForeground(false)` (default)
//! - Flat ball structuring element of radius r.
//!
//! # Complexity
//!
//! O(N) where N is the total voxel count, independent of the radius `r`.
//! A flat cubic structuring element is the Minkowski sum of three orthogonal
//! line segments, and dilation distributes over the Minkowski sum
//! (`D_{A⊕B} = D_A ∘ D_B`), so the `(2r+1)³` cube is computed as three separable
//! 1-D passes. Each pass is a linear nearest-foreground distance transform, so
//! total work is `O(3N)` rather than the direct `O(N · (2r+1)³)`.
//!
//! # References
//!
//! - Haralick, R.M., Sternberg, S.R., & Zhuang, X. (1987). Image analysis
//!   using mathematical morphology. *IEEE TPAMI*, 9(4), 532–550.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::types::ForegroundValue;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary dilation filter for 3-D images.
///
/// Grows foreground regions by one layer of `radius` voxels.  Each voxel is
/// foreground in the output iff at least one voxel in its `(2r+1)³` cubic
/// neighbourhood is foreground in the input.
///
/// Out-of-bounds positions are treated as background, so the foreground
/// region cannot grow beyond the image boundary.
#[derive(Debug, Clone)]
pub struct BinaryDilateFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryDilateFilter {
    /// Create a binary dilation filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary dilation to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let result = dilate_binary_3d(&vals, dims, self.radius, self.foreground_value);

        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for BinaryDilateFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Binary dilation on a flat Z×Y×X volume.
///
/// # Invariants
///
/// - Output length = `nz × ny × nx`.
/// - Output[i] ∈ {foreground_value, 0.0}.
/// - Output[i] = foreground_value iff any (2r+1)³ in-bounds neighbour = fg.
pub(crate) fn dilate_binary_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    fg: ForegroundValue,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let fg: f32 = fg.into();
    if n == 0 {
        return Vec::new();
    }

    // Work in a boolean buffer, then map back to {fg, 0.0}. Three separable
    // 1-D dilations (x, then y, then z) reproduce the (2r+1)³ cubic result
    // exactly: each pass marks a voxel foreground iff some in-bounds neighbour
    // within `radius` along that axis is foreground, and out-of-bounds positions
    // never contribute foreground — matching the cubic filter's boundary rule.
    let mut buf: Vec<bool> = data.iter().map(|&v| v == fg).collect();
    let mut scratch = vec![false; nx.max(ny).max(nz)];

    // X axis: contiguous runs of length nx, stride 1.
    for line in 0..(nz * ny) {
        dilate_line(&mut buf, line * nx, 1, nx, radius, &mut scratch);
    }
    // Y axis: runs of length ny, stride nx.
    for iz in 0..nz {
        for ix in 0..nx {
            dilate_line(&mut buf, iz * ny * nx + ix, nx, ny, radius, &mut scratch);
        }
    }
    // Z axis: runs of length nz, stride ny*nx.
    let plane = ny * nx;
    for iy in 0..ny {
        for ix in 0..nx {
            dilate_line(&mut buf, iy * nx + ix, plane, nz, radius, &mut scratch);
        }
    }

    buf.into_iter().map(|b| if b { fg } else { 0.0 }).collect()
}

/// In-place 1-D binary dilation of one strided line by a flat segment of
/// half-width `radius`. Output voxel `i` is foreground iff some in-bounds
/// position within `radius` of `i` is foreground in the input line. Linear in
/// the line length via a two-sided nearest-foreground sweep; `scratch` (length
/// ≥ `len`) is reused across calls to avoid per-line allocation.
fn dilate_line(
    buf: &mut [bool],
    start: usize,
    stride: usize,
    len: usize,
    radius: usize,
    scratch: &mut [bool],
) {
    let r = radius as isize;
    // Forward sweep: distance back to the most recent foreground voxel.
    let mut last_fg = isize::MIN;
    for i in 0..len {
        if buf[start + i * stride] {
            last_fg = i as isize;
        }
        scratch[i] = last_fg != isize::MIN && (i as isize - last_fg) <= r;
    }
    // Backward sweep: distance forward to the next foreground voxel.
    let mut next_fg = isize::MAX;
    for i in (0..len).rev() {
        if buf[start + i * stride] {
            next_fg = i as isize;
        }
        if next_fg != isize::MAX && (next_fg - i as isize) <= r {
            scratch[i] = true;
        }
    }
    for i in 0..len {
        buf[start + i * stride] = scratch[i];
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_dilate.rs"]
mod tests_binary_dilate;
