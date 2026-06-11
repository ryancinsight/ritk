//! Morphological Laplacian filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The morphological Laplacian is the difference between the dilated and
//! eroded versions of an image, scaled so that the original image cancels:
//!
//!   L_B(f) = (D_B f)(x) + (E_B f)(x) − 2 f(x)
//!
//! where:
//!
//!   (D_B f)(x) = max_{b ∈ B} f(x − b)
//!   (E_B f)(x) = min_{b ∈ B} f(x + b)
//!
//! and B is the cubic structuring element of half-width `radius`:
//!
//!   B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }
//!
//! # Relationship to scipy
//!
//! Mirrors `scipy.ndimage.morphological_laplace` with default arguments:
//!
//!   scipy.ndimage.morphological_laplace(f, size=2r+1, mode='reflect', cval=0.0)
//!
//!   tmp1 = scipy.ndimage.grey_dilation(f, size, mode='reflect', cval=0.0)
//!   tmp2 = scipy.ndimage.grey_erosion (f, size, mode='reflect', cval=0.0)
//!   out  = tmp1 + tmp2 − 2 f
//!
//! # Boundary Handling
//!
//! Half-sample symmetric reflection (scipy's `mode='reflect'`):
//!
//! The reflect extension has period `2 * n` and follows the pattern
//! `[0, 1, …, n−1, n−1, n−2, …, 1, 0, 0, 1, …]` — the edge value is
//! repeated once (no double repeat). For `n == 1` the only valid index
//! is 0, so every OOB index maps to 0.
//!
//! Note: this differs from the replicate (clamp) padding used by
//! [`GrayscaleDilation`](crate::morphology::GrayscaleDilation) and
//! [`GrayscaleErosion`](crate::morphology::GrayscaleErosion). The
//! reflect mode is required for byte-exact parity with scipy's default
//! `morphological_laplace`.
//!
//! # Properties
//!
//! - **Idempotence on constant fields**: L_B(c) = 0 for all constants c.
//! - **Translation invariance**: L_B(f(· − t))(x) = (L_B f)(x − t).
//! - **Increasing**: f ≤ g ⇒ L_B f ≤ L_B g (since both D and E are increasing).
//! - **Self-cancelling**: L_B(f) = 0 everywhere if f is locally constant.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) where N = n_z · n_y · n_x is the total voxel count
//! (3 passes: one dilation, one erosion, one elementwise combination).
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.
//! - scipy.ndimage.morphological_laplace:
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphological_laplace.html>

use ritk_core::filter::ops::extract_vec;
use ritk_core::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Morphological Laplacian filter for 3-D images.
///
/// Computes `L_B(f) = D_B(f) + E_B(f) − 2 f` using a cubic structuring
/// element of half-width `radius` and half-sample symmetric reflection
/// (scipy's `mode='reflect'`) at the boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MorphologicalLaplacian {
    radius: usize,
}

impl MorphologicalLaplacian {
    /// Construct a morphological Laplacian with cubic structuring element of
    /// half-width `radius`. The structuring element has side length
    /// `2 * radius + 1`; a value of `0` collapses to the 1×1×1 trivial
    /// element and yields the zero image.
    #[inline]
    pub const fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Half-width of the cubic structuring element.
    #[inline]
    pub const fn radius(&self) -> usize {
        self.radius
    }

    /// Apply the morphological Laplacian to a 3-D image.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let dilated = dilate_3d_reflect(&vals, dims, self.radius);
        let eroded = erode_3d_reflect(&vals, dims, self.radius);

        let mut out = Vec::with_capacity(vals.len());
        for ((&f, &d), &e) in vals.iter().zip(dilated.iter()).zip(eroded.iter()) {
            out.push(d + e - 2.0 * f);
        }

        let device = image.data().device();
        let out_td = TensorData::new(out, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Reflect an out-of-bounds index back into `[0, n)` using half-sample
/// symmetric reflection (scipy's `mode='reflect'`).
#[inline]
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * n as isize;
    let m = i.rem_euclid(period) as usize;
    if m < n {
        m
    } else {
        (2 * n) - m - 1
    }
}

/// Dilation with half-sample symmetric reflection at the boundary.
pub(crate) fn dilate_3d_reflect(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut max_val = f32::NEG_INFINITY;
                let pz = iz as isize;
                let py = iy as isize;
                let px = ix as isize;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = reflect_index(pz + dz, nz);
                            let yy = reflect_index(py + dy, ny);
                            let xx = reflect_index(px + dx, nx);
                            let val = data[zz * ny * nx + yy * nx + xx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }

                output[iz * ny * nx + iy * nx + ix] = max_val;
            }
        }
    }

    output
}

/// Erosion with half-sample symmetric reflection at the boundary.
pub(crate) fn erode_3d_reflect(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut min_val = f32::INFINITY;
                let pz = iz as isize;
                let py = iy as isize;
                let px = ix as isize;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = reflect_index(pz + dz, nz);
                            let yy = reflect_index(py + dy, ny);
                            let xx = reflect_index(px + dx, nx);
                            let val = data[zz * ny * nx + yy * nx + xx];
                            if val < min_val {
                                min_val = val;
                            }
                        }
                    }
                }

                output[iz * ny * nx + iy * nx + ix] = min_val;
            }
        }
    }

    output
}

#[cfg(test)]
mod tests;
