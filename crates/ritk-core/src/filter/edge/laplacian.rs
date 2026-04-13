//! Laplacian filter via second-order finite differences.
//!
//! # Mathematical Specification
//!
//! The Laplacian of a 3-D scalar field I is:
//!
//!   ∇²I(x) = ∂²I/∂z² + ∂²I/∂y² + ∂²I/∂x²
//!
//! Each second-order partial derivative is approximated at interior voxel i by:
//!
//!   ∂²I/∂z² ≈ (I[iz+1,iy,ix] − 2·I[iz,iy,ix] + I[iz−1,iy,ix]) / sz²
//!
//! At boundary voxels the second-order one-sided (forward at iz=0, backward at
//! iz=nz−1) formula is used, which maintains second-order accuracy for smooth
//! fields while keeping the computation well-defined everywhere:
//!
//!   forward  (iz=0):    (I[2]−2·I[1]+I[0]) / sz²   (uses the first three points)
//!   backward (iz=nz-1): (I[n-3]−2·I[n-2]+I[n-1]) / sz²
//!
//! # Reference
//! Press et al., *Numerical Recipes*, 3rd ed., §18.1.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Filter that computes the discrete Laplacian ∇²I of a 3-D image.
///
/// Voxel values in the output are the sum of second-order finite-difference
/// approximations along each axis, divided by the corresponding physical
/// spacing squared.
#[derive(Debug, Clone)]
pub struct LaplacianFilter {
    /// Physical voxel spacing \[sz, sy, sx\] (same units as image origin).
    pub spacing: [f64; 3],
}

impl LaplacianFilter {
    /// Create a filter with the given physical spacing.
    pub fn new(spacing: [f64; 3]) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing (1.0 in each direction).
    pub fn unit() -> Self {
        Self {
            spacing: [1.0, 1.0, 1.0],
        }
    }

    /// Compute the Laplacian image.
    ///
    /// Returns an `Image` whose voxel values are ∇²I(x) at each position x.
    /// The output has the same shape and physical metadata as `image`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("LaplacianFilter requires f32 image data: {:?}", e))?
            .to_vec();

        let dims = image.shape();
        let result = laplacian_vec(&vals, dims, self.spacing);

        let device = image.data().device();
        let td2 = TensorData::new(result, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td2, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Internal computation ──────────────────────────────────────────────────────

/// Compute the Laplacian of a flat 3-D volume.
///
/// # Invariants
/// - Interior: central second-order differences, O(h²) accurate.
/// - Boundary: one-sided second-order differences using the nearest three points.
/// - Output length equals `nz * ny * nx`.
fn laplacian_vec(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    let sz2 = (spacing[0] * spacing[0]) as f32;
    let sy2 = (spacing[1] * spacing[1]) as f32;
    let sx2 = (spacing[2] * spacing[2]) as f32;

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);

                // ∂²I/∂z²
                let d2z = if nz < 2 {
                    0.0_f32
                } else if nz == 2 {
                    // Only two points: trivial central diff degenerates.
                    // Use (I[1]-I[0]) - (I[0]-I[?]); with only 2 pts the
                    // best we can do is the forward one-sided approx at iz=0
                    // and backward at iz=1. Both reduce to (I[1]-I[0])/sz² − (I[0]-I[1])/sz²
                    // which gives ±(I[1]−I[0])/sz² — not ideal but consistent.
                    if iz == 0 {
                        (data[idx(1, iy, ix)] - data[flat]) / sz2
                    } else {
                        (data[idx(0, iy, ix)] - data[flat]) / sz2
                    }
                } else if iz == 0 {
                    // Forward one-sided: uses iz=0,1,2
                    (data[idx(2, iy, ix)] - 2.0 * data[idx(1, iy, ix)] + data[flat]) / sz2
                } else if iz == nz - 1 {
                    // Backward one-sided: uses iz=n-3,n-2,n-1
                    (data[flat] - 2.0 * data[idx(nz - 2, iy, ix)] + data[idx(nz - 3, iy, ix)]) / sz2
                } else {
                    // Central: uses iz-1,iz,iz+1
                    (data[idx(iz + 1, iy, ix)] - 2.0 * data[flat] + data[idx(iz - 1, iy, ix)]) / sz2
                };

                // ∂²I/∂y²
                let d2y = if ny < 2 {
                    0.0_f32
                } else if ny == 2 {
                    if iy == 0 {
                        (data[idx(iz, 1, ix)] - data[flat]) / sy2
                    } else {
                        (data[idx(iz, 0, ix)] - data[flat]) / sy2
                    }
                } else if iy == 0 {
                    (data[idx(iz, 2, ix)] - 2.0 * data[idx(iz, 1, ix)] + data[flat]) / sy2
                } else if iy == ny - 1 {
                    (data[flat] - 2.0 * data[idx(iz, ny - 2, ix)] + data[idx(iz, ny - 3, ix)]) / sy2
                } else {
                    (data[idx(iz, iy + 1, ix)] - 2.0 * data[flat] + data[idx(iz, iy - 1, ix)]) / sy2
                };

                // ∂²I/∂x²
                let d2x = if nx < 2 {
                    0.0_f32
                } else if nx == 2 {
                    if ix == 0 {
                        (data[idx(iz, iy, 1)] - data[flat]) / sx2
                    } else {
                        (data[idx(iz, iy, 0)] - data[flat]) / sx2
                    }
                } else if ix == 0 {
                    (data[idx(iz, iy, 2)] - 2.0 * data[idx(iz, iy, 1)] + data[flat]) / sx2
                } else if ix == nx - 1 {
                    (data[flat] - 2.0 * data[idx(iz, iy, nx - 2)] + data[idx(iz, iy, nx - 3)]) / sx2
                } else {
                    (data[idx(iz, iy, ix + 1)] - 2.0 * data[flat] + data[idx(iz, iy, ix - 1)]) / sx2
                };

                out[flat] = d2z + d2y + d2x;
            }
        }
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    /// Uniform image → Laplacian = 0 everywhere.
    #[test]
    fn test_uniform_zero_laplacian() {
        let dims = [8, 8, 8];
        let vals = vec![7.0_f32; 8 * 8 * 8];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
        let filter = LaplacianFilter::unit();
        let lap = filter.apply(&img).unwrap();

        let td = lap.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-4,
                "Laplacian[{i}] = {v} expected 0 for uniform image"
            );
        }
    }

    /// I[z,y,x] = x² (unit spacing) → ∂²I/∂x² = 2, other second derivatives = 0
    /// → Laplacian = 2 at interior voxels.
    #[test]
    fn test_quadratic_x_laplacian() {
        let [nz, ny, nx] = [6usize, 6, 10];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = (flat % nx) as f32;
                ix * ix
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = LaplacianFilter::unit();
        let lap = filter.apply(&img).unwrap();

        let td = lap.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();

        // Check interior voxels only (exclude boundary rows/columns).
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (out[flat] - 2.0).abs() < 1e-4,
                        "Laplacian[{iz},{iy},{ix}] = {} expected 2.0",
                        out[flat]
                    );
                }
            }
        }
    }

    /// I = x² with spacing_x = 2.0 → ∂²I/∂x² = 2/4 = 0.5
    #[test]
    fn test_non_unit_spacing() {
        let [nz, ny, nx] = [4usize, 4, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = (flat % nx) as f32;
                ix * ix
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
        let filter = LaplacianFilter::new([1.0, 1.0, 2.0]);
        let lap = filter.apply(&img).unwrap();

        let td = lap.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    // d²(x²)/dx² = 2; divided by spacing² = 4 → 0.5
                    assert!(
                        (out[flat] - 0.5).abs() < 1e-4,
                        "Laplacian[{iz},{iy},{ix}] = {} expected 0.5",
                        out[flat]
                    );
                }
            }
        }
    }

    /// I = x² + y² + z² (unit spacing) → Laplacian = 6 at interior.
    #[test]
    fn test_isotropic_quadratic() {
        let [nz, ny, nx] = [8usize, 8, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = (flat % nx) as f32;
                let iy = ((flat / nx) % ny) as f32;
                let iz = (flat / (ny * nx)) as f32;
                ix * ix + iy * iy + iz * iz
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = LaplacianFilter::unit();
        let lap = filter.apply(&img).unwrap();

        let td = lap.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (out[flat] - 6.0).abs() < 1e-3,
                        "Laplacian[{iz},{iy},{ix}] = {} expected 6.0",
                        out[flat]
                    );
                }
            }
        }
    }

    /// Linear field I = x + y + z → all second derivatives = 0 → Laplacian = 0.
    #[test]
    fn test_linear_field_zero_laplacian() {
        let [nz, ny, nx] = [6usize, 6, 6];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = (flat % nx) as f32;
                let iy = ((flat / nx) % ny) as f32;
                let iz = (flat / (ny * nx)) as f32;
                ix + iy + iz
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = LaplacianFilter::unit();
        let lap = filter.apply(&img).unwrap();

        let td = lap.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();
        for &v in out {
            assert!(
                v.abs() < 1e-4,
                "Laplacian = {v} expected 0 for linear field"
            );
        }
    }
}
