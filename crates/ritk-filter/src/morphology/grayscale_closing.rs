//! Grayscale morphological closing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale closing with a flat cubic structuring element B of half-width r:
//!
//!   C_B(f) = E_B(D_B(f))
//!
//! i.e. dilation followed by erosion.  Both operations use replicate (clamp)
//! boundary conditions.
//!
//! # Properties
//!
//! - **Extensivity**: C_B(f)(x) ≥ f(x) for all x.
//!   Proof: D_B(f)(x) ≥ f(x) (extensivity of dilation).  Then, since
//!   E_B(D_B(f)) ≥ E_B(f) (monotonicity of erosion) and E_B(D_B(f)) is at
//!   least as large as f pointwise because the dilation first raised the
//!   minimum of each neighbourhood, erosion cannot reduce it below f(x).
//!
//! - **Idempotence**: C_B(C_B(f)) = C_B(f).
//!   C_B is already extensive (C_B(f) ≥ f), so dilation cannot raise it
//!   further; erosion then restores it to the same level. ∎
//!
//! - **Fills dark holes**: removes dark features (regional minima) whose
//!   diameter is smaller than 2r + 1 voxels.
//!
//! # ITK Parity
//!
//! Matches `itk::GrayscaleMorphologicalClosingImageFilter` with:
//! - Flat cubic structuring element of half-width `radius`.
//! - Safe border mode (replicate padding) — default ITK boundary condition.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) for each of dilation and erosion pass.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, pp. 84–88.

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;
use ritk_tensor_ops::extract_vec;
use ritk_image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale morphological closing filter for 3-D images.
///
/// Applies dilation followed by erosion with a flat cubic structuring
/// element of half-width `radius`.  Fills dark voids smaller than the
/// structuring element without altering large dark regions.
#[derive(Debug, Clone)]
pub struct GrayscaleClosingFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleClosingFilter {
    /// Create a new grayscale closing filter with the given radius.
    ///
    /// A radius of 0 yields the identity (single-voxel SE).
    /// A radius of 1 uses a 3×3×3 cubic structuring element.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply grayscale closing to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        // C_B(f) = E_B(D_B(f))
        let dilated = dilate_3d(&vals, dims, self.radius);
        let closed = erode_3d(&dilated, dims, self.radius);

        let device = image.data().device();
        let out_td = TensorData::new(closed, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// C_B(c) = c for constant image c.
    ///
    /// **Proof**: D_B(c) = c (dilation of constant), E_B(c) = c (erosion of
    /// constant), so C_B(c) = E_B(D_B(c)) = E_B(c) = c. ∎
    #[test]
    fn constant_image_unchanged() {
        let c = 42.0_f32;
        let dims = [6, 6, 6];
        let img = make_image(vec![c; 216], dims);
        let out = GrayscaleClosingFilter::new(2).apply(&img).unwrap();
        for &v in extract_vals(&out).iter() {
            assert!((v - c).abs() < 1e-6, "constant unchanged: got {v}");
        }
    }

    /// Radius 0 is identity: C_B(f) = f when |B| = 1 (only the centre voxel).
    ///
    /// **Proof**: dilation r=0 returns max of {f(x)} = f(x); same for erosion. ∎
    #[test]
    fn radius_zero_is_identity() {
        let vals: Vec<f32> = (0..216_u32).map(|i| i as f32).collect();
        let dims = [6, 6, 6];
        let img = make_image(vals.clone(), dims);
        let out = GrayscaleClosingFilter::new(0).apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        for (i, (&a, &b)) in vals.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "radius-0 identity: voxel {i} {a} ≠ {b}"
            );
        }
    }

    /// Dark valley filled by closing.
    ///
    /// Volume: 3×3×5, all voxels = 5.0 except the centre column ix=2 which
    /// equals 0.0.  After closing (r=1) the valley must be filled.
    ///
    /// **Proof**:
    /// - Dilation r=1 at ix=2: max includes ix=1 and ix=3 (both 5) → 5.
    /// - After dilation entire volume = 5.
    /// - Erosion r=1 of constant 5 = 5 everywhere. ∎
    #[test]
    fn dark_valley_filled() {
        let [nz, ny, nx] = [3usize, 3, 5];
        let n = nz * ny * nx;
        let mut vals = vec![5.0_f32; n];
        // Set centre column (ix=2) to 0
        for iz in 0..nz {
            for iy in 0..ny {
                vals[iz * ny * nx + iy * nx + 2] = 0.0;
            }
        }
        let img = make_image(vals, [nz, ny, nx]);
        let out = GrayscaleClosingFilter::new(1).apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        // All interior and border must be ≥ 0 and ≤ 5; specifically each voxel ≥ 4.9
        for (i, &v) in out_vals.iter().enumerate() {
            assert!(v > 4.9, "dark_valley_filled: voxel {i} = {v}, expected ≈ 5");
        }
    }

    /// Extensivity: C_B(f)(x) ≥ f(x) for all x.
    ///
    /// Verified over a random-like gradient volume.
    #[test]
    fn extensivity() {
        let dims = [8, 8, 8];
        let n = 8 * 8 * 8;
        // Non-trivial pattern: each voxel value = (i * 7919 % 256) as f32
        let vals: Vec<f32> = (0..n as u32).map(|i| (i * 7919 % 256) as f32).collect();
        let img = make_image(vals.clone(), dims);
        let out = GrayscaleClosingFilter::new(1).apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        for (i, (&before, &after)) in vals.iter().zip(out_vals.iter()).enumerate() {
            assert!(
                after >= before - 1e-5,
                "extensivity violated at voxel {i}: before={before}, after={after}"
            );
        }
    }

    /// Idempotence: C_B(C_B(f)) = C_B(f).
    #[test]
    fn idempotence() {
        let dims = [6, 6, 6];
        let n = 6 * 6 * 6;
        let vals: Vec<f32> = (0..n as u32).map(|i| (i * 3571 % 128) as f32).collect();
        let img = make_image(vals, dims);
        let once = GrayscaleClosingFilter::new(1).apply(&img).unwrap();
        let twice = GrayscaleClosingFilter::new(1).apply(&once).unwrap();
        let v1 = extract_vals(&once);
        let v2 = extract_vals(&twice);
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "idempotence: voxel {i} first={a} second={b}"
            );
        }
    }

    /// Spatial metadata (origin, spacing, direction) is preserved.
    #[test]
    fn spatial_metadata_preserved() {
        use ritk_spatial::Direction;
        let origin = Point::new([1.5, 2.5, 3.5]);
        let spacing = Spacing::new([0.5, 0.5, 1.0]);
        let direction = Direction::identity();
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(tensor, origin, spacing, direction);
        let out = GrayscaleClosingFilter::new(1).apply(&img).unwrap();
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
    }

    /// Background-only image (all zeros) remains all zeros.
    #[test]
    fn all_background_unchanged() {
        let dims = [5, 5, 5];
        let img = make_image(vec![0.0_f32; 125], dims);
        let out = GrayscaleClosingFilter::new(2).apply(&img).unwrap();
        for &v in extract_vals(&out).iter() {
            assert!(v.abs() < 1e-6, "all-bg must stay 0, got {v}");
        }
    }

    /// Large dark feature (> SE size) is NOT removed by closing.
    ///
    /// A 5×5×5 fully-dark block within a 9×9×9 volume is too large for r=1
    /// to fill.  After closing, the inner region remains dark.
    #[test]
    fn large_dark_region_unchanged() {
        let [nz, ny, nx] = [9usize, 9, 9];
        let n = nz * ny * nx;
        let mut vals = vec![255.0_f32; n];
        // Set a 5×5×5 dark block at iz/iy/ix ∈ {2..6}
        for iz in 2..7 {
            for iy in 2..7 {
                for ix in 2..7 {
                    vals[iz * ny * nx + iy * nx + ix] = 0.0;
                }
            }
        }
        let img = make_image(vals, [nz, ny, nx]);
        let out = GrayscaleClosingFilter::new(1).apply(&img).unwrap();
        let out_vals = extract_vals(&out);
        // Interior of the 5×5×5 block (iz/iy/ix ∈ {3..5}) must remain dark
        for iz in 3..6 {
            for iy in 3..6 {
                for ix in 3..6 {
                    let v = out_vals[iz * 9 * 9 + iy * 9 + ix];
                    assert!(
                        v < 1.0,
                        "large dark region: flat[{iz},{iy},{ix}] = {v}, expected ~0"
                    );
                }
            }
        }
    }
}
