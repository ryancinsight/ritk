//! Grayscale erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale erosion with a flat structuring element B is defined as:
//!
//!   (E_B f)(x) = min_{b ∈ B} f(x + b)
//!
//! where B is a cubic neighbourhood of half-width `radius`:
//!
//!   B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }
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
//! - **Idempotence on constant fields**: E_B(c) = c for all constants c.
//! - **Anti-extensivity**: (E_B f)(x) ≤ f(x) for all x (with flat B containing
//!   the origin).
//! - **Translation invariance**: E_B(f(· − t))(x) = (E_B f)(x − t).
//! - **Increasing**: f ≤ g ⇒ E_B f ≤ E_B g.
//! - **Duality with dilation**: E_B f = −(D_B(−f)) for flat structuring elements.
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

/// Grayscale erosion filter for 3-D images.
///
/// Replaces each voxel with the minimum value in its `(2r+1)³` cubic
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
#[derive(Debug, Clone)]
pub struct GrayscaleErosion {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleErosion {
    /// Create a new grayscale erosion filter with the given radius.
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

    /// Apply grayscale erosion to a 3-D image.
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
            .map_err(|e| anyhow::anyhow!("GrayscaleErosion requires f32 data: {:?}", e))?
            .to_vec();

        let eroded = erode_3d(&vals, dims, self.radius);

        let device = image.data().device();
        let out_td = TensorData::new(eroded, Shape::new(dims));
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

/// Compute grayscale erosion on a flat 3-D volume stored in Z×Y×X order.
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
/// - Each output voxel equals `min_{b ∈ B} data[clamp(x + b)]`.
fn erode_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut min_val = f32::INFINITY;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                            let yy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                            let xx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
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

    /// Erosion of a constant image returns the same constant.
    ///
    /// **Proof**: min_{b ∈ B} c = c for all constants c. ∎
    #[test]
    fn test_constant_image_unchanged() {
        let dims = [8, 8, 8];
        let c = 7.0_f32;
        let vals = vec![c; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims);

        let filter = GrayscaleErosion::new(2);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - c).abs() < 1e-6,
                "constant image erosion: voxel {i} = {v}, expected {c}"
            );
        }
    }

    /// Erosion reduces a bright spot embedded in a dark background.
    ///
    /// A single bright voxel surrounded by a lower background value must be
    /// replaced by the background after erosion with radius ≥ 1, because the
    /// minimum over the neighbourhood includes the background.
    ///
    /// **Proof**: Let f(x₀) = h (bright), f(x) = bg for x ≠ x₀ with bg < h.
    /// For any y with ‖y − x₀‖_∞ ≤ r, the neighbourhood B(y) contains at
    /// least one voxel x ≠ x₀ (since |B| ≥ 27 > 1), so
    /// (E_B f)(y) ≤ bg < h. In particular, (E_B f)(x₀) = bg. ∎
    #[test]
    fn test_bright_spot_reduced() {
        let dims = [8, 8, 8];
        let bg = 1.0_f32;
        let bright = 100.0_f32;
        let n = dims[0] * dims[1] * dims[2];
        let mut vals = vec![bg; n];

        // Place bright spot at centre (4, 4, 4)
        let centre = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
        vals[centre] = bright;

        let img = make_image(vals, dims);
        let filter = GrayscaleErosion::new(1);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // The bright spot should be replaced by the background
        assert!(
            (out[centre] - bg).abs() < 1e-6,
            "bright spot should be eroded to background: got {}, expected {bg}",
            out[centre]
        );

        // Interior voxels away from the spot should remain at background
        for iz in 1..dims[0] - 1 {
            for iy in 1..dims[1] - 1 {
                for ix in 1..dims[2] - 1 {
                    let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                    assert!(
                        (out[flat] - bg).abs() < 1e-6,
                        "interior voxel ({iz},{iy},{ix}) = {}, expected {bg}",
                        out[flat]
                    );
                }
            }
        }
    }

    /// Radius-0 erosion is identity.
    ///
    /// **Proof**: B = {0}, so (E_B f)(x) = min{ f(x) } = f(x). ∎
    #[test]
    fn test_radius_zero_identity() {
        let dims = [6, 6, 6];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img = make_image(vals.clone(), dims);

        let filter = GrayscaleErosion::new(0);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "radius-0 identity: voxel {i} = {actual}, expected {expected}"
            );
        }
    }

    /// Anti-extensivity: (E_B f)(x) ≤ f(x) for all x when B contains the origin.
    ///
    /// **Proof**: 0 ∈ B ⇒ min_{b ∈ B} f(x+b) ≤ f(x+0) = f(x). ∎
    #[test]
    fn test_anti_extensivity() {
        let dims = [8, 8, 8];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| (i % 37) as f32 + 1.0).collect();
        let img = make_image(vals.clone(), dims);

        let filter = GrayscaleErosion::new(1);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, (&original, &eroded)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                eroded <= original + 1e-6,
                "anti-extensivity violated at voxel {i}: eroded = {eroded} > original = {original}"
            );
        }
    }

    /// Opening (erosion then dilation) removes small bright features.
    ///
    /// A small bright cube (smaller than the structuring element) embedded in a
    /// uniform background should be removed by opening = dilation(erosion(f)).
    /// The erosion shrinks the feature below existence, then dilation cannot
    /// recover it.
    #[test]
    fn test_opening_removes_small_bright_feature() {
        let dims = [16, 16, 16];
        let bg = 0.0_f32;
        let bright = 100.0_f32;
        let n = dims[0] * dims[1] * dims[2];
        let mut vals = vec![bg; n];

        // Place a single bright voxel at (8, 8, 8)
        let centre = 8 * dims[1] * dims[2] + 8 * dims[2] + 8;
        vals[centre] = bright;

        let img = make_image(vals, dims);

        // Erosion with radius 1
        let erosion = GrayscaleErosion::new(1);
        let eroded = erosion.apply(&img).unwrap();

        // Dilation with radius 1 (import from sibling module)
        let dilation = crate::filter::morphology::GrayscaleDilation::new(1);
        let opened = dilation.apply(&eroded).unwrap();
        let out = extract_vals(&opened);

        // The single bright voxel should be completely removed by opening
        assert!(
            (out[centre] - bg).abs() < 1e-6,
            "opening should remove single bright voxel: got {}, expected {bg}",
            out[centre]
        );

        // All interior voxels should be at background level
        let margin = 2;
        for iz in margin..dims[0] - margin {
            for iy in margin..dims[1] - margin {
                for ix in margin..dims[2] - margin {
                    let flat = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                    assert!(
                        (out[flat] - bg).abs() < 1e-6,
                        "opening: voxel ({iz},{iy},{ix}) = {}, expected {bg}",
                        out[flat]
                    );
                }
            }
        }
    }
}
