//! Laplacian of Gaussian (LoG) filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The Laplacian of Gaussian is defined as:
//!
//!   LoG(x) = ∇²G_σ * I = G_σ * ∇²I
//!
//! where G_σ is the Gaussian kernel with standard deviation σ and ∇² is the
//! Laplacian operator. By the commutativity of convolution and the linearity
//! of differentiation, the two orderings are equivalent.
//!
//! The closed-form 3-D LoG kernel is:
//!
//!   LoG(r) = −(1/(πσ⁴)) · [1 − r²/(2σ²)] · exp(−r²/(2σ²))
//!
//! where r² = x² + y² + z². This implementation uses the separable approach:
//! first apply Gaussian smoothing (via `GaussianFilter`), then compute the
//! discrete Laplacian (via `LaplacianFilter`). This reuses existing verified
//! components and avoids constructing a large 3-D kernel.
//!
//! # Properties
//!
//! - **LoG of a constant field is zero**: ∇²(constant) = 0.
//! - **Zero-crossing detection**: Edges correspond to zero crossings of the
//!   LoG response.
//! - **Blob detection**: The LoG response is negative at the centre of a
//!   bright Gaussian blob with matching scale, enabling blob detection via
//!   scale-space extrema.
//!
//! # Complexity
//!
//! O(N) for the Laplacian stage, plus the cost of the separable Gaussian
//! convolution (O(D · N · k) where k is the kernel half-width per dimension).
//!
//! # References
//!
//! - Marr, D. & Hildreth, E. (1980). Theory of edge detection. *Proceedings
//!   of the Royal Society of London B*, 207(1167), pp. 187–217.
//! - Lindeberg, T. (1994). *Scale-Space Theory in Computer Vision*. Springer.

use crate::filter::edge::LaplacianFilter;
use crate::filter::GaussianFilter;
use crate::image::Image;
use burn::tensor::backend::Backend;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Laplacian of Gaussian (LoG) filter for 3-D images.
///
/// Computes ∇²(G_σ * I) by first applying Gaussian smoothing with standard
/// deviation σ in each dimension (respecting physical spacing), then computing
/// the discrete Laplacian via second-order finite differences.
#[derive(Debug, Clone)]
pub struct LaplacianOfGaussianFilter {
    /// Standard deviation of the Gaussian in physical units (mm).
    sigma: f64,
}

impl LaplacianOfGaussianFilter {
    /// Create a new LoG filter with the given sigma (physical units).
    pub fn new(sigma: f64) -> Self {
        Self { sigma }
    }

    /// Set the Gaussian sigma.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Apply the LoG filter to a 3-D image.
    ///
    /// Computes G_σ * ∇²I by:
    /// 1. Smoothing the image with a Gaussian of standard deviation σ.
    /// 2. Computing the discrete Laplacian of the smoothed image.
    ///
    /// The output has the same shape and spatial metadata as the input.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as
    /// `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let spacing = image.spacing();
        let sp = [spacing[0], spacing[1], spacing[2]];

        // Stage 1: Gaussian smoothing
        let gauss = GaussianFilter::<B>::new(vec![self.sigma, self.sigma, self.sigma]);
        let smoothed = gauss.apply(image);

        // Stage 2: Laplacian via second-order finite differences
        let laplacian = LaplacianFilter::new(sp);
        laplacian.apply(&smoothed)
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

    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// LoG of a constant image is zero everywhere.
    ///
    /// **Proof**: Let I(x) = c for all x.
    ///   G_σ * I = c  (Gaussian integrates to 1).
    ///   ∇²(c) = 0.
    /// Therefore LoG(I) = 0. ∎
    #[test]
    fn test_constant_image_zero_log() {
        let dims = [16, 16, 16];
        let c = 42.0_f32;
        let vals = vec![c; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

        let filter = LaplacianOfGaussianFilter::new(1.5);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // Check interior voxels (boundary may have artefacts from the
        // Gaussian conv1d padding propagating into the Laplacian stencil)
        let margin = 6;
        let [nz, ny, nx] = dims;
        for iz in margin..nz - margin {
            for iy in margin..ny - margin {
                for ix in margin..nx - margin {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        out[flat].abs() < 0.1,
                        "LoG of constant image should be zero, but voxel ({iz},{iy},{ix}) = {}",
                        out[flat]
                    );
                }
            }
        }
    }

    /// LoG response at the centre of a bright Gaussian blob is negative.
    ///
    /// **Derivation**: A bright isotropic Gaussian blob I(r) = A·exp(−r²/(2σ_b²))
    /// has positive curvature (concave down) at its peak. The Laplacian of a
    /// Gaussian-smoothed version of this blob is negative at the centre for
    /// σ_smooth near σ_blob, because:
    ///
    ///   ∇²(G_σ * I)(0) < 0
    ///
    /// when I is a bright bump. This is the foundation of LoG-based blob
    /// detection (Lindeberg 1994).
    #[test]
    fn test_gaussian_blob_negative_centre() {
        let [nz, ny, nx] = [32usize, 32, 32];
        let n = nz * ny * nx;
        let cz = nz as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cx = nx as f64 / 2.0;
        let sigma_blob = 3.0_f64;
        let two_sigma2 = 2.0 * sigma_blob * sigma_blob;
        let amplitude = 100.0_f64;

        let vals: Vec<f32> = (0..n)
            .map(|flat| {
                let ix = flat % nx;
                let iy = (flat / nx) % ny;
                let iz = flat / (ny * nx);
                let dz = iz as f64 - cz;
                let dy = iy as f64 - cy;
                let dx = ix as f64 - cx;
                let r2 = dz * dz + dy * dy + dx * dx;
                (amplitude * (-r2 / two_sigma2).exp()) as f32
            })
            .collect();

        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

        // Use a sigma close to the blob scale for maximum LoG response
        let filter = LaplacianOfGaussianFilter::new(3.0);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // The centre voxel should have a negative LoG response
        let centre_flat = (nz / 2) * ny * nx + (ny / 2) * nx + (nx / 2);
        assert!(
            out[centre_flat] < 0.0,
            "LoG at centre of bright Gaussian blob should be negative, got {}",
            out[centre_flat]
        );

        // Verify the response is substantially negative (not just rounding noise)
        assert!(
            out[centre_flat] < -0.1,
            "LoG response at blob centre should be substantially negative, got {}",
            out[centre_flat]
        );
    }

    /// LoG of a linear field I = x + y + z is zero (∇² of a linear function = 0).
    ///
    /// **Proof**: G_σ * (ax + by + cz + d) = ax + by + cz + d (linear functions
    /// are invariant under Gaussian smoothing except at boundaries).
    /// ∇²(ax + by + cz + d) = 0. ∎
    #[test]
    fn test_linear_field_zero_log() {
        let [nz, ny, nx] = [16usize, 16, 16];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n)
            .map(|flat| {
                let ix = (flat % nx) as f32;
                let iy = ((flat / nx) % ny) as f32;
                let iz = (flat / (ny * nx)) as f32;
                ix + iy + iz
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

        let filter = LaplacianOfGaussianFilter::new(1.5);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // Interior voxels should be near zero
        let margin = 5;
        for iz in margin..nz - margin {
            for iy in margin..ny - margin {
                for ix in margin..nx - margin {
                    let flat = iz * ny * nx + iy * nx + ix;
                    assert!(
                        out[flat].abs() < 0.5,
                        "LoG of linear field should be ~0 at interior, but voxel ({iz},{iy},{ix}) = {}",
                        out[flat]
                    );
                }
            }
        }
    }
}
