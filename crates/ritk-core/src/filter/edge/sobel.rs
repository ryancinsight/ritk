//! 3-D Sobel gradient filter via separable convolution.
//!
//! # Mathematical Specification
//!
//! The 3-D Sobel operator estimates spatial derivatives using separable
//! convolution kernels that combine a derivative operator with smoothing.
//! For each axis direction, the Sobel kernel is the outer product of three
//! 1-D kernels:
//!
//!   d = \[-1, 0, 1\]  (derivative)
//!   s = \[ 1, 2, 1\]  (smoothing)
//!
//! For the x-derivative: K\_x = s ⊗ s ⊗ d  (smooth z, smooth y, derivative x)
//! For the y-derivative: K\_y = s ⊗ d ⊗ s  (smooth z, derivative y, smooth x)
//! For the z-derivative: K\_z = d ⊗ s ⊗ s  (derivative z, smooth y, smooth x)
//!
//! Each 3×3×3 kernel is applied via three sequential 1-D convolutions
//! with replicate (clamp) boundary padding.
//!
//! ## Normalization
//!
//! The raw convolution output is normalized to approximate the true spatial
//! gradient in physical units. The normalization factor for each component is:
//!
//!   factor = 2 · h · 4 · 4 = 32 · h
//!
//! where h is the physical spacing along the derivative axis. The factor of 2·h
//! accounts for the central-difference step size (the derivative kernel
//! \[-1, 0, 1\] computes f(i+1) − f(i−1), spanning 2 voxels), and each factor
//! of 4 is the sum of one smoothing kernel \[1, 2, 1\].
//!
//! ## Proof sketch (linear ramp)
//!
//! Let I(z, y, x) = x with unit spacing. At any interior voxel:
//!   1. Derivative along x: I(x+1) − I(x−1) = 2
//!   2. Smooth along y: \[1,2,1\] · \[2,2,2\] = 8
//!   3. Smooth along z: \[1,2,1\] · \[8,8,8\] = 32
//!   4. Normalize: 32 / (32 · 1.0) = 1.0  ✓  (true gradient of I = x is 1)
//!
//! ## Gradient Magnitude
//!
//!   |∇I| = √(G\_z² + G\_y² + G\_x²)
//!
//! ## Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to
//! \[0, dim\_size − 1\]. This yields one-sided differences at boundaries
//! (with halved magnitude relative to central differences).
//!
//! # Reference
//!
//! Zucker, S. W. & Hummel, R. A. (1981). "A three-dimensional edge operator."
//! *IEEE Trans. Pattern Analysis and Machine Intelligence*, 3(3), 324–331.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// 3-D Sobel gradient filter.
///
/// Computes spatial derivatives using the 3-D Sobel operator, which combines
/// central-difference derivative estimation with binomial smoothing along
/// the two orthogonal axes. The output is normalized to physical gradient
/// units (intensity per unit spacing).
///
/// ## Kernel structure
///
/// For derivative axis `a` with orthogonal axes `b`, `c`:
///
/// ```text
/// K_a[db][dc][da] = s[db] · s[dc] · d[da]
///   where d = [-1, 0, 1], s = [1, 2, 1]
/// ```
///
/// ## Normalization factor derivation
///
/// | Component        | Factor | Source                                      |
/// |------------------|--------|---------------------------------------------|
/// | Central diff     | 2·h    | \[-1,0,1\] spans 2 voxels of spacing h      |
/// | Smoothing axis 1 | 4      | sum(\[1,2,1\])                               |
/// | Smoothing axis 2 | 4      | sum(\[1,2,1\])                               |
/// | **Total**        | 32·h   |                                             |
#[derive(Debug, Clone)]
pub struct SobelFilter {
    /// Physical voxel spacing \[sz, sy, sx\].
    pub spacing: [f64; 3],
}

impl SobelFilter {
    /// Create a filter with the given physical spacing \[sz, sy, sx\].
    pub fn new(spacing: [f64; 3]) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing \[1.0, 1.0, 1.0\].
    pub fn unit() -> Self {
        Self {
            spacing: [1.0, 1.0, 1.0],
        }
    }

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I| = √(G\_z² + G\_y² + G\_x²).
    /// The output has the same shape and physical metadata (origin, spacing,
    /// direction) as the input.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = sobel_components(&vals, dims, self.spacing);
        let mag: Vec<f32> = gz
            .iter()
            .zip(gy.iter())
            .zip(gx.iter())
            .map(|((&z, &y), &x)| (z * z + y * y + x * x).sqrt())
            .collect();
        Ok(rebuild(mag, dims, image))
    }

    /// Compute the three gradient component images.
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape
    /// and physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<B, 3>,
    ) -> anyhow::Result<(Image<B, 3>, Image<B, 3>, Image<B, 3>)> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = sobel_components(&vals, dims, self.spacing);
        Ok((
            rebuild(gz, dims, image),
            rebuild(gy, dims, image),
            rebuild(gx, dims, image),
        ))
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract `(Vec<f32>, [nz, ny, nx])` from an `Image<B, 3>`.
fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let td = image.data().clone().into_data();
    let vals = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("SobelFilter requires f32 data: {:?}", e))?
        .to_vec();
    Ok((vals, image.shape()))
}

/// Rebuild an `Image<B, 3>` from a flat `Vec<f32>`, inheriting metadata from `src`.
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

/// Compute Sobel gradient components (gz, gy, gx) via separable 1-D convolutions.
///
/// For each component:
///   1. Apply derivative kernel \[-1, 0, 1\] along the target axis.
///   2. Apply smoothing kernel \[1, 2, 1\] along each orthogonal axis.
///   3. Normalize by 32 · h\_axis.
///
/// Boundary handling: replicate (clamp) padding.
///
/// # Invariants
///
/// - Interior voxels receive second-order central-difference gradient estimates.
/// - Output lengths equal `nz × ny × nx`.
/// - For a linear field I = c·x\_a, the interior component along axis a equals c,
///   and all orthogonal components equal zero.
fn sobel_components(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
    let smooth: [f32; 3] = [1.0, 2.0, 1.0];

    // G_z: derivative along z (axis 0), smooth along y (axis 1) and x (axis 2).
    let gz = {
        let tmp = convolve_1d_axis(data, dims, 0, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 1, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        let norm = 32.0 * spacing[0] as f32;
        normalize_vec(raw, norm)
    };

    // G_y: derivative along y (axis 1), smooth along z (axis 0) and x (axis 2).
    let gy = {
        let tmp = convolve_1d_axis(data, dims, 1, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        let norm = 32.0 * spacing[1] as f32;
        normalize_vec(raw, norm)
    };

    // G_x: derivative along x (axis 2), smooth along z (axis 0) and y (axis 1).
    let gx = {
        let tmp = convolve_1d_axis(data, dims, 2, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 1, &smooth);
        let norm = 32.0 * spacing[2] as f32;
        normalize_vec(raw, norm)
    };

    (gz, gy, gx)
}

/// Divide every element of `v` by `norm`.
#[inline]
fn normalize_vec(v: Vec<f32>, norm: f32) -> Vec<f32> {
    let inv = 1.0 / norm;
    v.into_iter().map(|x| x * inv).collect()
}

/// Apply a 3-tap 1-D convolution along the specified axis with replicate padding.
///
/// `axis`: 0 = z, 1 = y, 2 = x.
/// `kernel`: 3-element filter \[k\_{-1}, k\_0, k\_{+1}\].
///
/// Boundary indices are clamped to \[0, dim\_size − 1\] (replicate padding).
///
/// # Complexity
///
/// O(N) where N = nz × ny × nx. Each voxel performs exactly 3 multiply-adds.
fn convolve_1d_axis(data: &[f32], dims: [usize; 3], axis: usize, kernel: &[f32; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    let stride: usize = match axis {
        0 => ny * nx,
        1 => nx,
        2 => 1,
        _ => unreachable!(),
    };
    let dim_len = dims[axis];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let base = iz * ny * nx + iy * nx + ix;
                let pos = match axis {
                    0 => iz,
                    1 => iy,
                    2 => ix,
                    _ => unreachable!(),
                };

                let mut sum = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let offset = ki as isize - 1; // -1, 0, +1
                    let neighbor = (pos as isize + offset).clamp(0, dim_len as isize - 1) as usize;
                    let neighbor_flat = (base as isize
                        + (neighbor as isize - pos as isize) * stride as isize)
                        as usize;
                    sum += kv * data[neighbor_flat];
                }
                out[base] = sum;
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

    /// Construct a test image from flat data, dimensions, and spacing.
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

    /// Uniform image → gradient magnitude = 0.0 everywhere.
    ///
    /// Proof: derivative of a constant is identically zero. The derivative
    /// kernel [-1, 0, 1] applied to [c, c, c] yields 0 regardless of padding
    /// strategy. All subsequent smoothing passes preserve zero.
    #[test]
    fn test_uniform_image_zero_gradient() {
        let dims = [8, 8, 8];
        let vals = vec![42.0_f32; 8 * 8 * 8];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
        let filter = SobelFilter::unit();
        let mag = filter.apply(&img).unwrap();

        let td = mag.data().clone().into_data();
        let out = td.as_slice::<f32>().unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "magnitude[{i}] = {v}, expected 0.0 for uniform image"
            );
        }
    }

    /// Linear ramp I(z,y,x) = x with unit spacing.
    ///
    /// Expected at interior voxels:
    ///   gx = 1.0, gy = 0.0, gz = 0.0, magnitude = 1.0
    ///
    /// Derivation:
    ///   Derivative along x: (x+1)−(x−1) = 2
    ///   Smooth y: [1,2,1]·[2,2,2] = 8
    ///   Smooth z: [1,2,1]·[8,8,8] = 32
    ///   Normalized: 32 / (32·1) = 1.0
    ///
    ///   Derivative along y of I=x: I(y+1,x)−I(y−1,x) = x−x = 0 → gy = 0
    ///   Derivative along z of I=x: analogous → gz = 0
    #[test]
    fn test_ramp_x_unit_spacing() {
        let [nz, ny, nx] = [8usize, 8, 12];
        let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = SobelFilter::unit();
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        // Check interior voxels only (1-voxel margin from each face).
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        gz_vals[flat].abs() < 1e-5,
                        "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                        gz_vals[flat]
                    );
                    assert!(
                        gy_vals[flat].abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - 1.0).abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {}, expected 1.0",
                        gx_vals[flat]
                    );
                }
            }
        }

        // Verify magnitude at interior.
        let mag = filter.apply(&img).unwrap();
        let mag_data = mag.data().clone().into_data();
        let mag_vals = mag_data.as_slice::<f32>().unwrap();
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (mag_vals[flat] - 1.0).abs() < 1e-5,
                        "magnitude[{iz},{iy},{ix}] = {}, expected 1.0",
                        mag_vals[flat]
                    );
                }
            }
        }
    }

    /// Diagonal ramp I(z,y,x) = x + y + z with unit spacing.
    ///
    /// Expected at interior: each component = 1.0, magnitude = √3.
    ///
    /// Derivation (each axis identical by symmetry):
    ///   Derivative along a: (c_a+1)−(c_a−1) = 2
    ///   Two smoothing passes on constant 2 → 32
    ///   Normalized: 32 / (32·1) = 1.0
    ///   |∇I| = √(1²+1²+1²) = √3 ≈ 1.7320508
    #[test]
    fn test_diagonal_ramp_magnitude() {
        let [nz, ny, nx] = [10usize, 10, 10];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = flat % nx;
                let iy = (flat / nx) % ny;
                let iz = flat / (ny * nx);
                (iz + iy + ix) as f32
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = SobelFilter::unit();
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
                        "gz[{iz},{iy},{ix}] = {}, expected 1.0",
                        gz_vals[flat]
                    );
                    assert!(
                        (gy_vals[flat] - 1.0).abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {}, expected 1.0",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - 1.0).abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {}, expected 1.0",
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
                        (mag_vals[flat] - expected_mag).abs() < 1e-4,
                        "magnitude[{iz},{iy},{ix}] = {}, expected {expected_mag}",
                        mag_vals[flat]
                    );
                }
            }
        }
    }

    /// Non-unit spacing: I(z,y,x) = x, spacing\_x = 2.0.
    ///
    /// Expected at interior: gx = 1.0 / 2.0 = 0.5
    ///
    /// Derivation:
    ///   Raw convolution is identical to unit-spacing case (index-space): 32
    ///   Normalized: 32 / (32 · 2.0) = 0.5
    #[test]
    fn test_non_unit_spacing() {
        let [nz, ny, nx] = [6usize, 6, 10];
        let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
        let filter = SobelFilter::new([1.0, 1.0, 2.0]);
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        gz_vals[flat].abs() < 1e-5,
                        "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                        gz_vals[flat]
                    );
                    assert!(
                        gy_vals[flat].abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - 0.5).abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {}, expected 0.5",
                        gx_vals[flat]
                    );
                }
            }
        }

        // Verify magnitude = 0.5 at interior.
        let mag = filter.apply(&img).unwrap();
        let mag_data = mag.data().clone().into_data();
        let mag_vals = mag_data.as_slice::<f32>().unwrap();
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (mag_vals[flat] - 0.5).abs() < 1e-5,
                        "magnitude[{iz},{iy},{ix}] = {}, expected 0.5",
                        mag_vals[flat]
                    );
                }
            }
        }
    }

    /// Metadata preservation: origin, spacing, and direction pass through
    /// unmodified to all output images.
    #[test]
    fn test_metadata_preserved() {
        let dims = [4, 4, 4];
        let vals = vec![1.0_f32; 4 * 4 * 4];

        let origin = Point::new([10.0, -5.0, 3.5]);
        let spacing_val = Spacing::new([0.5, 1.5, 2.5]);
        let mut direction = Direction::identity();
        // 90-degree rotation around the z-axis (proper rotation, det = 1).
        direction[(0, 0)] = 0.0;
        direction[(0, 1)] = -1.0;
        direction[(1, 0)] = 1.0;
        direction[(1, 1)] = 0.0;

        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(tensor, origin, spacing_val, direction);

        let filter = SobelFilter::new([0.5, 1.5, 2.5]);

        // Check apply (magnitude).
        let mag = filter.apply(&img).unwrap();
        assert_eq!(mag.origin(), &origin);
        assert_eq!(mag.spacing(), &spacing_val);
        assert_eq!(mag.direction(), &direction);
        assert_eq!(mag.shape(), dims);

        // Check apply_components.
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();
        for (label, component) in [("gz", &gz), ("gy", &gy), ("gx", &gx)] {
            assert_eq!(component.origin(), &origin, "{label}: origin mismatch");
            assert_eq!(
                component.spacing(),
                &spacing_val,
                "{label}: spacing mismatch"
            );
            assert_eq!(
                component.direction(),
                &direction,
                "{label}: direction mismatch"
            );
            assert_eq!(component.shape(), dims, "{label}: shape mismatch");
        }
    }

    /// Y-axis ramp: I(z,y,x) = y with unit spacing.
    ///
    /// Verifies axis separation: gy = 1.0, gx = gz = 0 at interior.
    /// This confirms that the derivative and smoothing kernels are applied
    /// along the correct axes.
    #[test]
    fn test_ramp_y_axis_separation() {
        let [nz, ny, nx] = [8usize, 10, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let iy = (flat / nx) % ny;
                iy as f32
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = SobelFilter::unit();
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        gz_vals[flat].abs() < 1e-5,
                        "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                        gz_vals[flat]
                    );
                    assert!(
                        (gy_vals[flat] - 1.0).abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {}, expected 1.0",
                        gy_vals[flat]
                    );
                    assert!(
                        gx_vals[flat].abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {}, expected 0.0",
                        gx_vals[flat]
                    );
                }
            }
        }
    }

    /// Z-axis ramp: I(z,y,x) = z with unit spacing.
    ///
    /// Verifies gz = 1.0, gy = gx = 0 at interior.
    #[test]
    fn test_ramp_z_axis_separation() {
        let [nz, ny, nx] = [10usize, 8, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let iz = flat / (ny * nx);
                iz as f32
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
        let filter = SobelFilter::unit();
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (gz_vals[flat] - 1.0).abs() < 1e-5,
                        "gz[{iz},{iy},{ix}] = {}, expected 1.0",
                        gz_vals[flat]
                    );
                    assert!(
                        gy_vals[flat].abs() < 1e-5,
                        "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                        gy_vals[flat]
                    );
                    assert!(
                        gx_vals[flat].abs() < 1e-5,
                        "gx[{iz},{iy},{ix}] = {}, expected 0.0",
                        gx_vals[flat]
                    );
                }
            }
        }
    }

    /// Anisotropic spacing: I = z + y + x, spacing = [0.5, 1.0, 2.0].
    ///
    /// Expected interior:
    ///   gz = 1.0 / 0.5 = 2.0
    ///   gy = 1.0 / 1.0 = 1.0
    ///   gx = 1.0 / 2.0 = 0.5
    ///   |∇I| = √(4.0 + 1.0 + 0.25) = √5.25 ≈ 2.2912878
    #[test]
    fn test_anisotropic_spacing_diagonal() {
        let [nz, ny, nx] = [8usize, 8, 8];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|flat| {
                let ix = flat % nx;
                let iy = (flat / nx) % ny;
                let iz = flat / (ny * nx);
                (iz + iy + ix) as f32
            })
            .collect();
        let sp = [0.5, 1.0, 2.0];
        let img = make_image(vals, [nz, ny, nx], sp);
        let filter = SobelFilter::new(sp);
        let (gz, gy, gx) = filter.apply_components(&img).unwrap();

        let gz_data = gz.data().clone().into_data();
        let gz_vals = gz_data.as_slice::<f32>().unwrap();
        let gy_data = gy.data().clone().into_data();
        let gy_vals = gy_data.as_slice::<f32>().unwrap();
        let gx_data = gx.data().clone().into_data();
        let gx_vals = gx_data.as_slice::<f32>().unwrap();

        let expected_gz = 1.0_f32 / 0.5;
        let expected_gy = 1.0_f32 / 1.0;
        let expected_gx = 1.0_f32 / 2.0;
        let expected_mag =
            (expected_gz * expected_gz + expected_gy * expected_gy + expected_gx * expected_gx)
                .sqrt();

        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        (gz_vals[flat] - expected_gz).abs() < 1e-4,
                        "gz[{iz},{iy},{ix}] = {}, expected {expected_gz}",
                        gz_vals[flat]
                    );
                    assert!(
                        (gy_vals[flat] - expected_gy).abs() < 1e-4,
                        "gy[{iz},{iy},{ix}] = {}, expected {expected_gy}",
                        gy_vals[flat]
                    );
                    assert!(
                        (gx_vals[flat] - expected_gx).abs() < 1e-4,
                        "gx[{iz},{iy},{ix}] = {}, expected {expected_gx}",
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
                        (mag_vals[flat] - expected_mag).abs() < 1e-4,
                        "magnitude[{iz},{iy},{ix}] = {}, expected {expected_mag}",
                        mag_vals[flat]
                    );
                }
            }
        }
    }
}
