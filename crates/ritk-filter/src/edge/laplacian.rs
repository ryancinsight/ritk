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
//! ∂²I/∂z² ≈ (I\[iz+1,iy,ix\] − 2·I\[iz,iy,ix\] + I\[iz−1,iy,ix\]) / sz²
//!
//! At boundary voxels the same `[1, −2, 1]` stencil is evaluated with the
//! out-of-range neighbour clamped to the edge voxel (ZeroFluxNeumann boundary
//! condition); e.g. at iz=0 the lower neighbour is `I[0]`, giving
//! `(I[1] − I[0]) / sz²`. This reproduces ITK's `LaplacianImageFilter`, which
//! couples the Laplacian operator with `ZeroFluxNeumannBoundaryCondition`, to
//! within float rounding.
//!
//! # Reference
//! Press et al., *Numerical Recipes*, 3rd ed., §18.1; boundary handling per ITK
//! `itk::ZeroFluxNeumannBoundaryCondition`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Filter that computes the discrete Laplacian ∇²I of a 3-D image.
///
/// Voxel values in the output are the sum of second-order finite-difference
/// approximations along each axis, divided by the corresponding physical
/// spacing squared.
#[derive(Debug, Clone)]
pub struct LaplacianFilter {
    /// Physical voxel spacing [sz, sy, sx] (same units as image origin).
    pub spacing: Spacing<3>,
}

impl LaplacianFilter {
    /// Create a filter with the given physical spacing.
    pub fn new(spacing: Spacing<3>) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing (1.0 in each direction).
    pub fn unit() -> Self {
        Self {
            spacing: Spacing::uniform(1.0),
        }
    }

    /// Compute the Laplacian image.
    ///
    /// Returns an `Image` whose voxel values are ∇²I(x) at each position x.
    /// The output has the same shape and physical metadata as `image`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = laplacian_vec(&vals, dims, &self.spacing);

        Ok(rebuild(result, dims, image))
    }
}

// ── Internal computation ──────────────────────────────────────────────────────

/// Compute the Laplacian of a flat 3-D volume.
///
/// # Invariants
/// - `[1, −2, 1]` second-difference stencil divided by spacing² along each axis.
/// - Boundary voxels clamp the out-of-range neighbour to the edge voxel
///   (ZeroFluxNeumann), matching ITK; a length-1 axis contributes 0.
/// - Output length equals `nz * ny * nx`.
fn laplacian_vec(data: &[f32], dims: [usize; 3], spacing: &Spacing<3>) -> Vec<f32> {
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
                let center = data[flat];

                // ZeroFluxNeumann: clamp out-of-range neighbours to the edge.
                let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
                let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
                let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));

                let d2z = (data[idx(zhi, iy, ix)] - 2.0 * center + data[idx(zlo, iy, ix)]) / sz2;
                let d2y = (data[idx(iz, yhi, ix)] - 2.0 * center + data[idx(iz, ylo, ix)]) / sy2;
                let d2x = (data[idx(iz, iy, xhi)] - 2.0 * center + data[idx(iz, iy, xlo)]) / sx2;

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
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_tensor_ops::extract_vec_infallible;

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

        let (out, _) = extract_vec_infallible(&lap);
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

        let (out, _) = extract_vec_infallible(&lap);

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
        let filter = LaplacianFilter::new([1.0, 1.0, 2.0].into());
        let lap = filter.apply(&img).unwrap();

        let (out, _) = extract_vec_infallible(&lap);

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

        let (out, _) = extract_vec_infallible(&lap);

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

    /// Linear field I = x + y + z → interior second derivatives = 0 → Laplacian = 0.
    ///
    /// The boundary is intentionally excluded: under the ZeroFluxNeumann boundary
    /// condition (ITK's convention, which this filter matches) a min-face voxel
    /// evaluates `(I[1] − 2·I[0] + I[0])/h² = (I[1] − I[0])/h² = slope/h²`, which
    /// is nonzero for a non-constant linear field. Asserting zero there would be
    /// analytically incorrect for the boundary condition under test.
    #[test]
    fn test_linear_field_zero_laplacian_interior() {
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

        let (out, _) = extract_vec_infallible(&lap);
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let v = out[iz * ny * nx + iy * nx + ix];
                    assert!(
                        v.abs() < 1e-4,
                        "interior Laplacian[{iz},{iy},{ix}] = {v} expected 0 for linear field"
                    );
                }
            }
        }
    }
}
