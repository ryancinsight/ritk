//! Gradient magnitude filter using central finite differences.
//!
//! # Mathematical Specification
//!
//! For a 3-D image I defined on a regular grid with physical spacing (sz, sy, sx),
//! the gradient at interior voxel (iz, iy, ix) is estimated by central differences:
//!
//!   ∂I/∂z ≈ (I[iz+1, iy, ix] − I[iz−1, iy, ix]) / (2 · sz)
//!   ∂I/∂y ≈ (I[iz, iy+1, ix] − I[iz, iy−1, ix]) / (2 · sy)
//!   ∂I/∂x ≈ (I[iz, iy, ix+1] − I[iz, iy, ix−1]) / (2 · sx)
//!
//! At boundary voxels one-sided first-order differences are used:
//!   forward:  (I[i+1] − I[i]) / s
//!   backward: (I[i] − I[i−1]) / s
//!
//! Gradient magnitude: |∇I| = √(gz² + gy² + gx²)
//!
//! # Reference
//! Standard finite difference approximation of the gradient (see e.g., Press et al.,
//! *Numerical Recipes in C*, 3rd ed., §18.1).

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Filter that computes the gradient magnitude of a 3-D image.
///
/// All gradient components are divided by the corresponding physical spacing so
/// that the result is in units of intensity per millimetre (or whatever unit the
/// spacing is expressed in).
#[derive(Debug, Clone)]
pub struct GradientMagnitudeFilter {
    /// Physical voxel spacing [sz, sy, sx] in each axis direction.
    pub spacing: [f64; 3],
}

impl GradientMagnitudeFilter {
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

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I(x)| at each position x.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = gradient_vecs(&vals, dims, self.spacing);
        let mag: Vec<f32> = (0..vals.len())
            .map(|i| (gz[i] * gz[i] + gy[i] * gy[i] + gx[i] * gx[i]).sqrt())
            .collect();
        Ok(rebuild(mag, dims, image))
    }

    /// Compute the three gradient component images (z, y, x).
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape and
    /// physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<B, 3>,
    ) -> anyhow::Result<(Image<B, 3>, Image<B, 3>, Image<B, 3>)> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = gradient_vecs(&vals, dims, self.spacing);
        Ok((
            rebuild(gz, dims, image),
            rebuild(gy, dims, image),
            rebuild(gx, dims, image),
        ))
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract `(Vec<f32>, [nz,ny,nx])` from an Image<B,3>.
fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let td = image.data().clone().into_data();
    let vals = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("GradientMagnitudeFilter requires f32 data: {:?}", e))?
        .to_vec();
    Ok((vals, image.shape()))
}

/// Rebuild an `Image<B,3>` from a flat `Vec<f32>`, inheriting metadata from `src`.
fn rebuild<B: Backend>(vals: Vec<f32>, dims: [usize; 3], src: &Image<B, 3>) -> Image<B, 3> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        src.origin().clone(),
        src.spacing().clone(),
        src.direction().clone(),
    )
}

/// Compute gradient component vectors (gz, gy, gx) via finite differences.
///
/// # Invariants
/// - Interior voxels: second-order central differences divided by 2·spacing.
/// - Boundary voxels: first-order one-sided differences divided by spacing.
/// - Output lengths equal `nz * ny * nx`.
fn gradient_vecs(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut gz = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];
    let mut gx = vec![0.0_f32; n];

    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);

                // ∂I/∂z
                gz[flat] = if nz == 1 {
                    0.0
                } else if iz == 0 {
                    (data[idx(1, iy, ix)] - data[flat]) / sz
                } else if iz == nz - 1 {
                    (data[flat] - data[idx(nz - 2, iy, ix)]) / sz
                } else {
                    (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) / (2.0 * sz)
                };

                // ∂I/∂y
                gy[flat] = if ny == 1 {
                    0.0
                } else if iy == 0 {
                    (data[idx(iz, 1, ix)] - data[flat]) / sy
                } else if iy == ny - 1 {
                    (data[flat] - data[idx(iz, ny - 2, ix)]) / sy
                } else {
                    (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) / (2.0 * sy)
                };

                // ∂I/∂x
                gx[flat] = if nx == 1 {
                    0.0
                } else if ix == 0 {
                    (data[idx(iz, iy, 1)] - data[flat]) / sx
                } else if ix == nx - 1 {
                    (data[flat] - data[idx(iz, iy, nx - 2)]) / sx
                } else {
                    (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) / (2.0 * sx)
                };
            }
        }
    }

    (gz, gy, gx)
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

    /// Uniform image → gradient magnitude = 0 everywhere.
    #[test]
    fn test_uniform_image_zero_gradient() {
        let dims = [8, 8, 8];
        let vals = vec![5.0_f32; 8 * 8 * 8];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
        let filter = GradientMagnitudeFilter::unit();
        let mag = filter.apply(&img).unwrap();

        let td = mag.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();
        for &v in out {
            assert!(v.abs() < 1e-5, "expected 0.0 for uniform image, got {v}");
        }
    }

    /// I[z,y,x] = x (unit spacing) → gx = 1.0 (interior), gy = gz = 0; magnitude = 1.0 (interior).
    #[test]
    fn test_ramp_x_gradient() {
        let [nz, ny, nx] = [6usize, 6, 10];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = flat % nx;
                ix as f32
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = GradientMagnitudeFilter::unit();
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        // Interior voxels: ix in 1..nx-1
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        gz_vals[flat].abs() < 1e-5,
                        "gz[{iz},{iy},{ix}] = {} expected 0.0",
                        gz_vals[flat]
                    );
                    assert!(
                        gy_vals[flat].abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {} expected 0.0",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - 1.0).abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {} expected 1.0",
                        gx_vals[flat]
                    );
                }
            }
        }

        // Magnitude image interior
        let mag = filter.apply(&img).unwrap();
        let mag_data = mag.data().clone().into_data();
        let mag_vals = mag_data.as_slice::<f32>().unwrap();
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (mag_vals[flat] - 1.0).abs() < 1e-5,
                        "magnitude[{iz},{iy},{ix}] = {} expected 1.0",
                        mag_vals[flat]
                    );
                }
            }
        }
    }

    /// I[z,y,x] = x + y + z (unit spacing) → each component = 1.0 (interior);
    /// magnitude = √3 ≈ 1.7320508 (interior).
    #[test]
    fn test_diagonal_ramp_gradient() {
        let [nz, ny, nx] = [8usize, 8, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = flat % nx;
                let iy = (flat / nx) % ny;
                let iz = flat / (ny * nx);
                (iz + iy + ix) as f32
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = GradientMagnitudeFilter::unit();
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        let expected_mag = 3.0_f32.sqrt();
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (gz_vals[flat] - 1.0).abs() < 1e-5,
                        "gz interior expected 1.0, got {}",
                        gz_vals[flat]
                    );
                    assert!(
                        (gy_vals[flat] - 1.0).abs() < 1e-5,
                        "gy interior expected 1.0, got {}",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - 1.0).abs() < 1e-5,
                        "gx interior expected 1.0, got {}",
                        gx_vals[flat]
                    );
                }
            }
        }

        let mag = filter.apply(&img).unwrap();
        let mag_data = mag.data().clone().into_data();
        let mag_vals = mag_data.as_slice::<f32>().unwrap();
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (mag_vals[flat] - expected_mag).abs() < 1e-5,
                        "magnitude interior expected √3≈{expected_mag}, got {}",
                        mag_vals[flat]
                    );
                }
            }
        }
    }

    /// Non-unit spacing: I[z,y,x] = x, spacing_x = 2.0 → gx = 0.5 (interior).
    #[test]
    fn test_non_unit_spacing() {
        let [nz, ny, nx] = [4usize, 4, 8];
        let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
        let filter = GradientMagnitudeFilter::new([1.0, 1.0, 2.0]);
        let (_, _, gx) = filter.apply_components(&img).unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();
        // interior gx = 1 pixel / 2.0 mm = 0.5
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (gx_vals[flat] - 0.5).abs() < 1e-5,
                        "gx with spacing=2.0 expected 0.5, got {}",
                        gx_vals[flat]
                    );
                }
            }
        }
    }
}
