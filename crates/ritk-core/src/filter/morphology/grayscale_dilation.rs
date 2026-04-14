//! Grayscale dilation filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale dilation with a flat structuring element B is defined as:
//!
//!   (D_B f)(x) = max_{b ∈ B} f(x - b)
//!
//! where B is a cubic neighbourhood of half-width `radius`:
//!
//!   B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }
//!
//! For a symmetric (origin-centred) structuring element, f(x - b) and f(x + b)
//! range over the same set, so the definition simplifies to:
//!
//!   (D_B f)(x) = max_{b ∈ B} f(x + b)
//!
//! giving |B| = (2r + 1)³ voxels per neighbourhood.
//!
//! # Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to the nearest
//! valid index along each axis, equivalent to extending the boundary voxels
//! infinitely outward.
//!
//! # Properties
//!
//! - **Idempotence on constant fields**: D_B(c) = c for all constants c.
//! - **Extensivity**: (D_B f)(x) ≥ f(x) for all x (with flat B containing
//!   the origin).
//! - **Translation invariance**: D_B(f(· − t))(x) = (D_B f)(x − t).
//! - **Increasing**: f ≤ g ⇒ D_B f ≤ D_B g.
//! - **Duality with erosion**: D_B f = −(E_B(−f)) for flat structuring elements.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) where N = n_z · n_y · n_x is the total voxel count.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale dilation filter for 3-D images.
///
/// Replaces each voxel with the maximum value in its `(2r+1)³` cubic
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
#[derive(Debug, Clone)]
pub struct GrayscaleDilation {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleDilation {
    /// Create a new grayscale dilation filter with the given radius.
    ///
    /// A radius of 0 yields identity (each voxel is its own sole neighbour).
    /// A radius of 1 produces a 3×3×3 cubic structuring element.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply grayscale dilation to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("GrayscaleDilation requires f32 data: {:?}", e))?
            .to_vec();

        let dilated = dilate_3d(&vals, dims, self.radius);

        let device = image.data().device();
        let out_td = TensorData::new(dilated, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute grayscale dilation on a flat 3-D volume stored in Z×Y×X order.
///
/// # Arguments
///
/// * `data`   — flat voxel values in row-major (Z-major) order.
/// * `dims`   — `[nz, ny, nx]`.
/// * `radius` — structuring element half-width in voxels.
///
/// # Invariants
///
/// - Output length equals `nz * ny * nx`.
/// - Each output voxel equals `max_{b ∈ B} data[clamp(x + b)]`.
fn dilate_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut max_val = f32::NEG_INFINITY;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                            let yy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                            let xx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
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

    /// Dilation of a constant image returns the same constant.
    ///
    /// **Proof**: max_{b ∈ B} c = c for all constants c. ∎
    #[test]
    fn test_constant_image_unchanged() {
        let dims = [8, 8, 8];
        let c = 7.0_f32;
        let vals = vec![c; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims);

        let filter = GrayscaleDilation::new(2);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - c).abs() < 1e-6,
                "constant image dilation: voxel {i} = {v}, expected {c}"
            );
        }
    }

    /// Dilation expands a bright spot: a single bright voxel at the centre
    /// should propagate to all voxels within the structuring element radius.
    ///
    /// **Proof**: Let f(x₀) = h, f(x) = bg for x ≠ x₀ with h > bg.
    /// For any y with ‖y − x₀‖_∞ ≤ r, x₀ ∈ B(y), so
    /// (D_B f)(y) ≥ f(x₀) = h > bg. For y with ‖y − x₀‖_∞ > r,
    /// B(y) ∩ {x₀} = ∅, so (D_B f)(y) = bg. ∎
    #[test]
    fn test_bright_spot_expanded() {
        let dims = [16, 16, 16];
        let bg = 1.0_f32;
        let bright = 100.0_f32;
        let n = dims[0] * dims[1] * dims[2];
        let mut vals = vec![bg; n];

        // Place bright spot at centre (8, 8, 8)
        let cz = 8usize;
        let cy = 8usize;
        let cx = 8usize;
        let centre = cz * dims[1] * dims[2] + cy * dims[2] + cx;
        vals[centre] = bright;

        let img = make_image(vals, dims);
        let radius = 1usize;
        let filter = GrayscaleDilation::new(radius);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // All voxels within L∞ distance ≤ radius of the centre should be bright
        for iz in 0..dims[0] {
            for iy in 0..dims[1] {
                for ix in 0..dims[2] {
                    let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                    let dz = (iz as isize - cz as isize).unsigned_abs();
                    let dy = (iy as isize - cy as isize).unsigned_abs();
                    let dx = (ix as isize - cx as isize).unsigned_abs();
                    let linf = dz.max(dy).max(dx);

                    if linf <= radius {
                        assert!(
                            (out[flat] - bright).abs() < 1e-6,
                            "dilation should expand bright spot: voxel ({iz},{iy},{ix}) = {}, expected {bright}",
                            out[flat]
                        );
                    } else if linf > radius + 1 {
                        // Voxels far from the spot remain at background
                        assert!(
                            (out[flat] - bg).abs() < 1e-6,
                            "voxel ({iz},{iy},{ix}) should remain at background: got {}, expected {bg}",
                            out[flat]
                        );
                    }
                }
            }
        }
    }

    /// Radius-0 dilation is identity.
    ///
    /// **Proof**: B = {0}, so (D_B f)(x) = max{ f(x) } = f(x). ∎
    #[test]
    fn test_radius_zero_identity() {
        let dims = [6, 6, 6];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img = make_image(vals.clone(), dims);

        let filter = GrayscaleDilation::new(0);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "radius-0 identity: voxel {i} = {actual}, expected {expected}"
            );
        }
    }

    /// Extensivity: (D_B f)(x) ≥ f(x) for all x when B contains the origin.
    ///
    /// **Proof**: 0 ∈ B ⇒ max_{b ∈ B} f(x+b) ≥ f(x+0) = f(x). ∎
    #[test]
    fn test_extensivity() {
        let dims = [8, 8, 8];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| (i % 37) as f32 + 1.0).collect();
        let img = make_image(vals.clone(), dims);

        let filter = GrayscaleDilation::new(1);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, (&original, &dilated)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                dilated >= original - 1e-6,
                "extensivity violated at voxel {i}: dilated = {dilated} < original = {original}"
            );
        }
    }

    /// Duality: D_B(f) = −E_B(−f) for flat symmetric structuring elements.
    ///
    /// Verify numerically that dilation of f equals the negation of erosion
    /// applied to −f.
    #[test]
    fn test_duality_with_erosion() {
        let dims = [8, 8, 8];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 41) as f32).collect();
        let neg_vals: Vec<f32> = vals.iter().map(|&v| -v).collect();

        let img = make_image(vals, dims);
        let neg_img = make_image(neg_vals, dims);

        let radius = 1;
        let dilation = GrayscaleDilation::new(radius);
        let erosion = crate::filter::morphology::GrayscaleErosion::new(radius);

        let dilated = extract_vals(&dilation.apply(&img).unwrap());
        let eroded_neg = extract_vals(&erosion.apply(&neg_img).unwrap());

        for i in 0..n {
            let expected = -eroded_neg[i];
            assert!(
                (dilated[i] - expected).abs() < 1e-5,
                "duality violated at voxel {i}: D_B(f) = {}, -E_B(-f) = {expected}",
                dilated[i]
            );
        }
    }
}
