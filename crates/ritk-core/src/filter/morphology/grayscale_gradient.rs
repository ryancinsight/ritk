//! Grayscale morphological gradient filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The morphological gradient (Beucher gradient) is defined as the difference
//! between grayscale dilation and grayscale erosion:
//!
//!   `grad_B(f)(x) = D_B(f)(x) − E_B(f)(x)`
//!
//! where `B` is a flat cubic structuring element of half-width `r`:
//!
//!   `B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }`
//!
//! and:
//!   - `D_B(f)(x) = max_{b ∈ B} f(x + b)` (dilation — replicate padding)
//!   - `E_B(f)(x) = min_{b ∈ B} f(x + b)` (erosion — replicate padding)
//!
//! Since `D_B(f)(x) ≥ f(x) ≥ E_B(f)(x)` for all x (extensivity and
//! anti-extensivity with flat origin-containing SE), the gradient is
//! non-negative everywhere:
//!
//!   `grad_B(f)(x) ≥ 0`
//!
//! # Properties
//!
//! - **Non-negativity**: `grad_B(f)(x) ≥ 0`.
//! - **Constant field**: `grad_B(c) = 0` for any constant c.
//! - **Edge detection**: `grad_B(f)` is large near morphological boundaries
//!   and small in smooth regions.
//! - **radius = 0**: SE = {0}, so `D_B = E_B = identity` ⟹ gradient = 0 everywhere.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) where N = n_z · n_y · n_x is the total voxel count.
//!
//! # References
//!
//! - Beucher, S. & Lantuéjoul, C. (1979). Use of watersheds in contour detection.
//!   In *International Workshop on Image Processing*.
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, §4.3.
//! - ITK `itk::GrayscaleMorphologicalGradientImageFilter`.

use crate::filter::ops::extract_vec;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale morphological gradient filter for 3-D images.
///
/// Computes `D_B(f) − E_B(f)` — the Beucher gradient — which highlights
/// morphological edges in a grayscale volume.
///
/// The output is non-negative everywhere and is zero on regions where the
/// image is locally constant within the structuring element's footprint.
///
/// # Example (conceptual)
/// A sharp step edge (0 → 10) with `radius = 1` produces gradient = 10 at the
/// boundary voxel (dilation = 10, erosion = 0) and gradient = 0 one voxel away
/// from the edge (dilation = erosion = same constant).
#[derive(Debug, Clone)]
pub struct GrayscaleMorphologicalGradientFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleMorphologicalGradientFilter {
    /// Create a new gradient filter with the given structuring element radius.
    ///
    /// `radius = 0` ⟹ structuring element = {0} ⟹ gradient = 0 everywhere
    /// (degenerate identity case).
    /// `radius = 1` ⟹ 3×3×3 cubic SE (standard morphological gradient).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius and return the modified filter.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply the morphological gradient to a 3-D image.
    ///
    /// Returns a new image with the same shape and spatial metadata as the
    /// input. All output voxel values are ≥ 0.
    ///
    /// # Errors
    /// Returns an error if the image data cannot be converted to `f32` (only
    /// possible with non-f32 backends).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let dilated = dilate_3d(&vals, dims, self.radius);
        let eroded = erode_3d(&vals, dims, self.radius);

        // gradient(x) = dilation(x) - erosion(x); ≥ 0 by extensivity / anti-extensivity.
        let gradient: Vec<f32> = dilated
            .into_iter()
            .zip(eroded)
            .map(|(d, e)| d - e)
            .collect();

        let device = image.data().device();
        let td = TensorData::new(gradient, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
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
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_vec()
    }

    /// Constant image → gradient = 0 everywhere.
    ///
    /// # Proof
    /// D_B(c)(x) = max_{b ∈ B} c = c.
    /// E_B(c)(x) = min_{b ∈ B} c = c.
    /// grad(x) = c − c = 0.
    #[test]
    fn constant_image_zero_gradient() {
        let img = make_image(vec![5.0; 27], [3, 3, 3]);
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply(&img)
            .unwrap();
        for &v in vals(&out).iter() {
            assert_eq!(
                v, 0.0_f32,
                "constant image must yield zero gradient; got {v}"
            );
        }
    }

    /// radius = 0 → gradient = 0 everywhere (degenerate SE = {0}).
    ///
    /// # Proof
    /// SE = {0}; D_B(f)(x) = f(x); E_B(f)(x) = f(x); grad = 0.
    #[test]
    fn radius_zero_always_zero() {
        let img = make_image(vec![0.0, 5.0, 10.0, 3.0, 8.0, 1.0], [1, 2, 3]);
        let out = GrayscaleMorphologicalGradientFilter::new(0)
            .apply(&img)
            .unwrap();
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "radius=0 must yield zero gradient; got {v}");
        }
    }

    /// Output is non-negative everywhere.
    ///
    /// # Proof
    /// D_B(f)(x) ≥ f(x) ≥ E_B(f)(x) (extensivity / anti-extensivity of flat SE).
    /// Therefore grad(x) = D_B(f)(x) − E_B(f)(x) ≥ 0.
    #[test]
    fn output_nonnegative_everywhere() {
        // Arbitrary non-constant volume.
        let data: Vec<f32> = (0..27).map(|i| (i as f32) * 3.7 - 20.0).collect();
        let img = make_image(data, [3, 3, 3]);
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply(&img)
            .unwrap();
        for &v in vals(&out).iter() {
            assert!(v >= 0.0, "gradient must be non-negative; got {v}");
        }
    }

    /// Step-edge volume: gradient = 10 at boundary, 0 away from it.
    ///
    /// # Analytical derivation
    /// Volume 1×3×7 with values [0,0,0, 10,10,10,10] (step at column 3).
    /// radius = 1, SE = [-1, 0, +1] along each axis; only the X axis matters here.
    ///
    /// For boundary voxel at x=3 (value 10): neighbourhood = {x=2, x=3, x=4};
    ///   dilation = max(0, 10, 10) = 10; erosion = min(0, 10, 10) = 0; grad = 10.
    /// For boundary voxel at x=2 (value 0): neighbourhood = {x=1, x=2, x=3};
    ///   dilation = max(0, 0, 10) = 10; erosion = min(0, 0, 10) = 0; grad = 10.
    /// For interior voxel at x=0 (value 0): neighbourhood = {x=0, x=0, x=1} (clamped);
    ///   dilation = 0; erosion = 0; grad = 0.
    /// For interior voxel at x=6 (value 10): neighbourhood = {x=5, x=6, x=6} (clamped);
    ///   dilation = 10; erosion = 10; grad = 0.
    #[test]
    fn step_edge_gradient_at_boundary() {
        let data = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let img = make_image(data, [1, 1, 7]);
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply(&img)
            .unwrap();
        let v = vals(&out);
        // Interior voxels far from the edge: gradient = 0
        assert_eq!(v[0], 0.0_f32, "voxel at x=0 (far left): gradient must be 0");
        assert_eq!(
            v[6], 0.0_f32,
            "voxel at x=6 (far right): gradient must be 0"
        );
        // Boundary voxels at x=2 and x=3
        assert_eq!(
            v[2], 10.0_f32,
            "boundary voxel x=2: dilation=10, erosion=0 → gradient=10"
        );
        assert_eq!(
            v[3], 10.0_f32,
            "boundary voxel x=3: dilation=10, erosion=0 → gradient=10"
        );
    }

    /// Spatial metadata is preserved.
    #[test]
    fn spatial_metadata_preserved() {
        let sp = Spacing::new([2.5, 3.5, 4.5]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, 2.0, 3.0], Shape::new([1usize, 1, 3]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply(&img)
            .unwrap();
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Single bright voxel in a dark volume: gradient is non-zero only in
    /// the neighbourhood of the bright voxel.
    ///
    /// # Derivation
    /// Volume 1×5×5, all voxels = 0 except center (2,2) = 100.
    /// With radius = 1:
    ///   - The bright voxel and its 8 2D-neighbours (plus 1 above/below in z=0) have
    ///     dilation = 100 (center in neighbourhood) and erosion = 0 (0 in neighbourhood).
    ///     gradient = 100.
    ///   - Corner/edge voxels ≥ 2 away from the center: both dilation and erosion
    ///     are 0 (the bright voxel is not in the 3×3×3 neighbourhood); gradient = 0.
    #[test]
    fn single_bright_voxel_gradient_ring() {
        let mut data = vec![0.0_f32; 25]; // 1×5×5
        data[12] = 100.0; // center voxel at (0, 2, 2)
        let img = make_image(data, [1, 5, 5]);
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply(&img)
            .unwrap();
        let v = vals(&out);
        // Corners of the 5×5 image (flat z=0): indices 0, 4, 20, 24 are ≥ 2 away
        // from center (2,2) in the x dimension — no contact with the bright voxel.
        assert_eq!(
            v[0], 0.0_f32,
            "corner (0,0): far from bright voxel, gradient must be 0"
        );
        assert_eq!(
            v[4], 0.0_f32,
            "corner (0,4): far from bright voxel, gradient must be 0"
        );
        assert_eq!(
            v[20], 0.0_f32,
            "corner (4,0): far from bright voxel, gradient must be 0"
        );
        assert_eq!(
            v[24], 0.0_f32,
            "corner (4,4): far from bright voxel, gradient must be 0"
        );
        // Center voxel itself: in its own neighbourhood → dilation=100, erosion=0; grad=100.
        assert_eq!(
            v[12], 100.0_f32,
            "center bright voxel: gradient must be 100"
        );
    }
}
