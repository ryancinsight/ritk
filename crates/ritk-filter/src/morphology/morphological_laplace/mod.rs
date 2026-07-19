//! Morphological Laplacian filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The morphological Laplacian is the difference between the dilated and
//! eroded versions of an image, scaled so that the original image cancels:
//!
//!   L_B(f) = (D_B f)(x) + (E_B f)(x) âˆ’ 2 f(x)
//!
//! where:
//!
//!   (D_B f)(x) = max_{b âˆˆ B} f(x âˆ’ b)
//!   (E_B f)(x) = min_{b âˆˆ B} f(x + b)
//!
//! and B is the cubic structuring element of half-width `radius`:
//!
//!   B = { b âˆˆ â„¤Â³ : |b_i| â‰¤ r  for i âˆˆ {0, 1, 2} }
//!
//! # Relationship to scipy
//!
//! Mirrors `scipy.ndimage.morphological_laplace` with default arguments:
//!
//!   scipy.ndimage.morphological_laplace(f, size=2r+1, mode='reflect', cval=0.0)
//!
//!   tmp1 = scipy.ndimage.grey_dilation(f, size, mode='reflect', cval=0.0)
//!   tmp2 = scipy.ndimage.grey_erosion (f, size, mode='reflect', cval=0.0)
//!   out  = tmp1 + tmp2 âˆ’ 2 f
//!
//! # Boundary Handling
//!
//! Half-sample symmetric reflection (scipy's `mode='reflect'`):
//!
//! The reflect extension has period `2 * n` and follows the pattern
//! `[0, 1, â€¦, nâˆ’1, nâˆ’1, nâˆ’2, â€¦, 1, 0, 0, 1, â€¦]` â€” the edge value is
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
//! - **Translation invariance**: L_B(f(Â· âˆ’ t))(x) = (L_B f)(x âˆ’ t).
//! - **Increasing**: f â‰¤ g â‡’ L_B f â‰¤ L_B g (since both D and E are increasing).
//! - **Self-cancelling**: L_B(f) = 0 everywhere if f is locally constant.
//!
//! # Complexity
//!
//! O(N Â· (2r + 1)Â³) where N = n_z Â· n_y Â· n_x is the total voxel count
//! (3 passes: one dilation, one erosion, one elementwise combination).
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.
//! - scipy.ndimage.morphological_laplace:
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphological_laplace.html>

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Morphological Laplacian filter for 3-D images.
///
/// Computes `L_B(f) = D_B(f) + E_B(f) âˆ’ 2 f` using a cubic structuring
/// element of half-width `radius` and half-sample symmetric reflection
/// (scipy's `mode='reflect'`) at the boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MorphologicalLaplacian {
    radius: usize,
}

impl MorphologicalLaplacian {
    /// Construct a morphological Laplacian with cubic structuring element of
    /// half-width `radius`. The structuring element has side length
    /// `2 * radius + 1`; a value of `0` collapses to the 1Ã—1Ã—1 trivial
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = laplace_vec(&vals, dims, self.radius);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`MorphologicalLaplacian::apply`].
    ///
    /// Runs the identical `dilate + erode âˆ’ 2Â·f` (reflect-boundary cubic SE) via
    /// the shared `laplace_vec` host core on the image's contiguous host
    /// buffer, so the result is bitwise-identical to the Burn path. No Burn
    /// tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let radius = self.radius;
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            laplace_vec(vals, dims, radius)
        })
    }
}

/// Substrate-agnostic host core for [`MorphologicalLaplacian`].
///
/// `L_B(f) = D_B(f) + E_B(f) âˆ’ 2Â·f` over the reflect-boundary cubic SE of the
/// given `radius` (scipy `mode='reflect'` parity). Zero on locally-constant
/// regions.
pub(crate) fn laplace_vec(vals: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let dilated = dilate_3d_reflect(vals, dims, radius);
    let eroded = erode_3d_reflect(vals, dims, radius);
    vals.iter()
        .zip(dilated.iter())
        .zip(eroded.iter())
        .map(|((&f, &d), &e)| d + e - 2.0 * f)
        .collect()
}

// â”€â”€ Core computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
///
/// PERF-378-02: each output voxel reads its neighbourhood from `data` (read-only)
/// with no inter-voxel write dependency â€” parallelised over the flat voxel index.
pub(crate) fn dilate_3d_reflect(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let slab = ny * nx;
    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny * nx, |flat| {
        let iz = (flat / slab) as isize;
        let iy = ((flat / nx) % ny) as isize;
        let ix = (flat % nx) as isize;
        let mut max_val = f32::NEG_INFINITY;
        for dz in -r..=r {
            let zz = reflect_index(iz + dz, nz);
            for dy in -r..=r {
                let yy = reflect_index(iy + dy, ny);
                for dx in -r..=r {
                    let xx = reflect_index(ix + dx, nx);
                    let val = data[zz * slab + yy * nx + xx];
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
        }
        max_val
    })
}

/// Erosion with half-sample symmetric reflection at the boundary.
///
/// PERF-378-02: mirror of `dilate_3d_reflect` â€” per-voxel min-fold over neighbourhood;
/// read-only input; parallelised over the flat voxel index.
pub(crate) fn erode_3d_reflect(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let slab = ny * nx;
    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny * nx, |flat| {
        let iz = (flat / slab) as isize;
        let iy = ((flat / nx) % ny) as isize;
        let ix = (flat % nx) as isize;
        let mut min_val = f32::INFINITY;
        for dz in -r..=r {
            let zz = reflect_index(iz + dz, nz);
            for dy in -r..=r {
                let yy = reflect_index(iy + dy, ny);
                for dx in -r..=r {
                    let xx = reflect_index(ix + dx, nx);
                    let val = data[zz * slab + yy * nx + xx];
                    if val < min_val {
                        min_val = val;
                    }
                }
            }
        }
        min_val
    })
}

#[cfg(test)]
mod tests;
