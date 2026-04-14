//! Edge-preserving bilateral filter for 3-D volumes.
//!
//! # Algorithm
//! For each centre voxel **p** the output is the weighted average of all
//! voxels **q** inside the axis-aligned cube `[p ± r]³`, where
//! `r = ⌈3 · σ_s⌉`:
//!
//! ```text
//! w(p, q) = exp(−‖p − q‖² / (2 σ_s²)) · exp(−(I(p) − I(q))² / (2 σ_r²))
//! Output(p) = Σ w(p,q) · I(q)  /  Σ w(p,q)
//! ```
//!
//! Out-of-bounds neighbours are **skipped** (only in-bounds voxels contribute
//! to numerator and denominator), so the estimator remains unbiased at image
//! boundaries.
//!
//! # Precision
//! All weight accumulation is performed in `f64` to avoid catastrophic
//! cancellation.
//!
//! # Complexity
//! O(n · (2r+1)³) per image, where `r = ⌈3 · σ_s⌉`.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Edge-preserving bilateral filter for 3-D volumes.
///
/// Combines a spatial Gaussian and an intensity-range Gaussian to smooth
/// homogeneous regions while preserving edges.
///
/// # Invariants
/// - `spatial_sigma` and `range_sigma` are clamped to a minimum of `1e-10`
///   before use, preventing division by zero.
/// - The neighbourhood radius is `⌈3 · spatial_sigma⌉` voxels.
/// - Accumulation uses `f64` arithmetic.
pub struct BilateralFilter {
    /// Spatial Gaussian sigma in voxels.
    pub spatial_sigma: f64,
    /// Intensity-range Gaussian sigma (same units as voxel values).
    pub range_sigma: f64,
}

impl BilateralFilter {
    /// Construct a new bilateral filter.
    ///
    /// # Arguments
    /// * `spatial_sigma` — standard deviation of the spatial Gaussian (voxels).
    /// * `range_sigma`   — standard deviation of the intensity Gaussian.
    pub fn new(spatial_sigma: f64, range_sigma: f64) -> Self {
        Self {
            spatial_sigma,
            range_sigma,
        }
    }

    /// Apply the filter to a 3-D image.
    ///
    /// Returns a new `Image` with identical shape and spatial metadata
    /// (origin, spacing, direction).
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (data, dims) = extract_vec(image)?;
        let filtered = bilateral_3d(&data, dims, self.spatial_sigma, self.range_sigma);
        Ok(rebuild(filtered, dims, image))
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract flat `Vec<f32>` and `[Z, Y, X]` shape from an image.
fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let td = image.data().clone().into_data();
    let vals = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("tensor as_slice<f32> failed: {e:?}"))?
        .to_vec();
    Ok((vals, image.shape()))
}

/// Rebuild an `Image` from flat voxel data, preserving the reference metadata.
fn rebuild<B: Backend>(vals: Vec<f32>, dims: [usize; 3], reference: &Image<B, 3>) -> Image<B, 3> {
    let device = reference.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        reference.origin().clone(),
        reference.spacing().clone(),
        reference.direction().clone(),
    )
}

/// Bilateral filter on a 3-D volume stored in flat Z×Y×X order.
///
/// # Algorithm
/// For each centre voxel **p**:
/// 1. Neighbourhood radius `r = ⌈3 · σ_s⌉`.
/// 2. For each neighbour **q** in `[p ± r]³` (out-of-bounds skipped):
///    `w(p, q) = exp(−d_s² / (2 σ_s²)) · exp(−d_r² / (2 σ_r²))`
///    where `d_s = ‖p − q‖`, `d_r = |I(p) − I(q)|`.
/// 3. `Output(p) = Σ w·I(q) / Σ w`.
///
/// Accumulation is f64 to avoid catastrophic cancellation.
fn bilateral_3d(data: &[f32], dims: [usize; 3], spatial_sigma: f64, range_sigma: f64) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);

    // Guard degenerate sigma values.
    let spatial_sigma = spatial_sigma.max(1e-10);
    let range_sigma = range_sigma.max(1e-10);

    let r = (3.0 * spatial_sigma).ceil() as isize;
    let inv_two_ss2 = 1.0_f64 / (2.0 * spatial_sigma * spatial_sigma);
    let inv_two_sr2 = 1.0_f64 / (2.0 * range_sigma * range_sigma);

    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let center_flat = iz * ny * nx + iy * nx + ix;
                let center_val = data[center_flat] as f64;

                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;

                for dz in -r..=r {
                    let nz_i = iz as isize + dz;
                    if nz_i < 0 || nz_i >= nz as isize {
                        continue;
                    }
                    for dy in -r..=r {
                        let ny_i = iy as isize + dy;
                        if ny_i < 0 || ny_i >= ny as isize {
                            continue;
                        }
                        for dx in -r..=r {
                            let nx_i = ix as isize + dx;
                            if nx_i < 0 || nx_i >= nx as isize {
                                continue;
                            }

                            let n_flat =
                                nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                            let n_val = data[n_flat] as f64;

                            // Spatial distance squared (voxel units).
                            let spatial_d2 = (dz * dz + dy * dy + dx * dx) as f64;
                            // Range distance squared.
                            let range_d2 = (center_val - n_val) * (center_val - n_val);

                            let w = (-spatial_d2 * inv_two_ss2 - range_d2 * inv_two_sr2).exp();

                            weighted_sum += w * n_val;
                            weight_total += w;
                        }
                    }
                }

                output[center_flat] = if weight_total > 1e-20 {
                    (weighted_sum / weight_total) as f32
                } else {
                    data[center_flat]
                };
            }
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    /// Construct a test image from flat values and shape `[Z, Y, X]`.
    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::<3>::new([1.0, 2.0, 3.0]),
            Spacing::<3>::new([0.5, 0.75, 1.25]),
            Direction::<3>::identity(),
        )
    }

    /// Extract flat `Vec<f32>` from an image (test utility).
    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── 1. Uniform image → unchanged ─────────────────────────────────────

    /// A constant image has zero range differences everywhere, so the
    /// bilateral filter reduces to a spatial Gaussian average of identical
    /// values.  Output must equal the constant.
    #[test]
    fn test_bilateral_uniform_image_unchanged() {
        let dims = [6, 8, 10];
        let val = 7.5_f32;
        let vals = vec![val; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims);

        let filter = BilateralFilter::new(1.5, 10.0);
        let out = filter.apply(&img).unwrap();

        let result = extract_vals(&out);
        assert_eq!(result.len(), dims[0] * dims[1] * dims[2]);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - val).abs() < 1e-5, "voxel {i}: expected {val}, got {v}");
        }
    }

    // ── 2. Edge preservation ─────────────────────────────────────────────

    /// Step edge along the X axis: left half = 20, right half = 200.
    /// With a tight range sigma the bilateral filter should NOT blur across
    /// the edge.  Voxels far from the boundary must remain near their
    /// original value.
    #[test]
    fn test_bilateral_edge_preservation() {
        let [nz, ny, nx] = [8usize, 8, 16];
        let n = nz * ny * nx;
        let mut vals = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    vals[iz * ny * nx + iy * nx + ix] = if ix < nx / 2 { 20.0 } else { 200.0 };
                }
            }
        }
        let img = make_image(vals, [nz, ny, nx]);

        // Tight range sigma → intensity difference across edge (180) ≫ σ_r
        // so cross-edge weights are negligible.
        let filter = BilateralFilter::new(1.0, 5.0);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        // Check voxels well inside each region (≥2 voxels from boundary).
        for iz in 0..nz {
            for iy in 0..ny {
                // Left interior.
                for ix in 0..(nx / 2 - 2) {
                    let v = result[iz * ny * nx + iy * nx + ix];
                    assert!(
                        (v - 20.0).abs() < 2.0,
                        "left interior voxel [{iz},{iy},{ix}]: expected ~20, got {v}"
                    );
                }
                // Right interior.
                for ix in (nx / 2 + 2)..nx {
                    let v = result[iz * ny * nx + iy * nx + ix];
                    assert!(
                        (v - 200.0).abs() < 2.0,
                        "right interior voxel [{iz},{iy},{ix}]: expected ~200, got {v}"
                    );
                }
            }
        }
    }

    // ── 3. Metadata preserved ────────────────────────────────────────────

    /// Origin, spacing, and direction of the output image must match the
    /// input exactly.
    #[test]
    fn test_bilateral_metadata_preserved() {
        let dims = [4, 4, 4];
        let vals = vec![1.0_f32; 64];
        let img = make_image(vals, dims);

        let filter = BilateralFilter::new(1.0, 1.0);
        let out = filter.apply(&img).unwrap();

        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        assert_eq!(out.shape(), img.shape());
    }

    // ── 4. Smooth region is smoothed ─────────────────────────────────────

    /// In a uniform region with additive noise, the bilateral filter should
    /// reduce variance while keeping the mean approximately unchanged.
    ///
    /// Construction: 8×8×8 image with base value 100.  A deterministic
    /// noise pattern (±5 alternating) is added.  After bilateral filtering
    /// with a large range sigma (noise amplitude ≪ σ_r), variance must
    /// decrease.
    #[test]
    fn test_bilateral_smooth_region_is_smoothed() {
        let [nz, ny, nx] = [8usize, 8, 8];
        let n = nz * ny * nx;
        let base = 100.0_f32;
        let noise_amp = 5.0_f32;

        // Deterministic alternating noise: +5 / -5 in a checkerboard.
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let iz = i / (ny * nx);
                let iy = (i / nx) % ny;
                let ix = i % nx;
                let sign = if (iz + iy + ix) % 2 == 0 { 1.0 } else { -1.0 };
                base + noise_amp * sign
            })
            .collect();

        let input_mean = vals.iter().sum::<f32>() / n as f32;
        let input_var = vals
            .iter()
            .map(|&v| (v - input_mean) * (v - input_mean))
            .sum::<f32>()
            / n as f32;

        let img = make_image(vals, [nz, ny, nx]);

        // Large range sigma so noise is within the range kernel → smoothed.
        let filter = BilateralFilter::new(1.5, 50.0);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        let output_mean = result.iter().sum::<f32>() / n as f32;
        let output_var = result
            .iter()
            .map(|&v| (v - output_mean) * (v - output_mean))
            .sum::<f32>()
            / n as f32;

        // Mean should be approximately conserved.
        assert!(
            (output_mean - input_mean).abs() < 1.0,
            "mean shifted: input={input_mean:.4} output={output_mean:.4}"
        );

        // Variance must decrease.
        assert!(
            output_var < input_var,
            "variance not reduced: input={input_var:.4} output={output_var:.4}"
        );

        // Quantitative check: variance should drop substantially.
        assert!(
            output_var < input_var * 0.5,
            "variance reduction insufficient: input={input_var:.4} output={output_var:.4}"
        );
    }
}
