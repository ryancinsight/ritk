//! Sliding-window median filter for 3-D volumes.
//!
//! # Algorithm
//! For each voxel `(iz, iy, ix)`, collect all values in the axis-aligned cube
//! `[iz ± r, iy ± r, ix ± r]` using replicate (clamp) boundary conditions,
//! sort them, and take the middle element (lower median for even-length
//! neighbourhoods, consistent with standard medical-imaging toolkits).
//!
//! # Complexity
//! O(n · (2r+1)³ · log((2r+1)³)) where n is the total voxel count and r is
//! the neighbourhood half-width. For r = 1 this is O(27 n log 27).

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Sliding-window median filter for 3-D volumes.
///
/// Replaces each voxel with the median of its `(2r+1)³` axis-aligned
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
pub struct MedianFilter {
    /// Neighbourhood half-width in voxels (default 1 → 3×3×3 cube).
    pub radius: usize,
}

impl MedianFilter {
    /// Create a new median filter with the given neighbourhood half-width.
    ///
    /// A radius of 0 yields identity (each voxel is its own sole neighbour).
    /// A radius of 1 produces a 3×3×3 kernel (27 samples per voxel).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply the median filter to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let shape = image.shape();
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("failed to extract f32 slice: {e:?}"))?
            .to_vec();

        let filtered = median_3d(&vals, shape, self.radius);

        let device = image.data().device();
        let out_td = TensorData::new(filtered, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);

        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

/// Sliding-window median on a 3-D volume stored in flat Z×Y×X order.
///
/// # Arguments
/// * `data`   — flat voxel values in row-major (Z-major) order.
/// * `dims`   — `[nz, ny, nx]`.
/// * `radius` — neighbourhood half-width in voxels.
///
/// # Boundary handling
/// Replicate (clamp) padding: out-of-bounds indices are clamped to the
/// nearest valid index along each axis.
fn median_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let r = radius as isize;
    let cap = (2 * radius + 1).pow(3);
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut neighbors: Vec<f32> = Vec::with_capacity(cap);

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            // Replicate (clamp) padding.
                            let zz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                            let yy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                            let xx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            neighbors.push(data[zz * ny * nx + yy * nx + xx]);
                        }
                    }
                }

                neighbors
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Lower median for even-length neighbourhoods.
                output[iz * ny * nx + iy * nx + ix] = neighbors[neighbors.len() / 2];
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

    /// Construct a test image from flat values, shape, and optional metadata.
    fn make_image(
        vals: Vec<f32>,
        dims: [usize; 3],
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    /// Extract voxel data as `Vec<f32>` from an image.
    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Test 1: Uniform image is unchanged ────────────────────────────────

    /// A constant image must be invariant under median filtering for any
    /// radius, because the median of identical values equals that value.
    ///
    /// **Proof sketch**: Let I(p) = c for all p. For any neighbourhood N(p),
    /// every element of the sorted list is c, so median = c. ∎
    #[test]
    fn test_uniform_image_unchanged() {
        let dims = [8, 8, 8];
        let val = 42.0_f32;
        let vals = vec![val; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);

        let filter = MedianFilter::new(2);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        assert_eq!(out.shape(), dims);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-6,
                "voxel {i}: expected {val}, got {v}"
            );
        }
    }

    // ── Test 2: Impulse noise removed ─────────────────────────────────────

    /// A single spike voxel (salt-noise) embedded in a constant field must be
    /// eliminated by median filtering with radius ≥ 1, because the spike
    /// constitutes at most 1 out of (2r+1)³ ≥ 27 samples and therefore
    /// cannot be the median of the sorted neighbourhood.
    ///
    /// **Proof**: In a 3×3×3 neighbourhood around the spike, 26 values equal
    /// the background `c` and 1 equals `c + spike`. Sorted, the 14th element
    /// (index 13) is `c`. ∎
    #[test]
    fn test_impulse_noise_removed() {
        let dims = [8, 8, 8];
        let bg = 10.0_f32;
        let spike = 1000.0_f32;
        let n = dims[0] * dims[1] * dims[2];
        let mut vals = vec![bg; n];

        // Place spike at centre voxel (4, 4, 4).
        let spike_idx = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
        vals[spike_idx] = spike;

        let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);
        let filter = MedianFilter::new(1);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        // The spike location must now hold the background value.
        assert!(
            (result[spike_idx] - bg).abs() < 1e-6,
            "spike not removed: expected {bg}, got {}",
            result[spike_idx]
        );

        // All other voxels in the interior (away from boundaries) must remain
        // at the background value since their entire neighbourhood is constant.
        for iz in 1..dims[0] - 1 {
            for iy in 1..dims[1] - 1 {
                for ix in 1..dims[2] - 1 {
                    let idx = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                    if idx == spike_idx {
                        continue;
                    }
                    // Voxels adjacent to the spike see at most 1 non-bg value
                    // out of 27, so their median is still bg.
                    assert!(
                        (result[idx] - bg).abs() < 1e-6,
                        "interior voxel ({iz},{iy},{ix}): expected {bg}, got {}",
                        result[idx]
                    );
                }
            }
        }
    }

    // ── Test 3: Metadata preserved ────────────────────────────────────────

    /// Origin, spacing, and direction of the output image must be identical
    /// to the input. The median filter operates exclusively on voxel
    /// intensities and must not mutate spatial metadata.
    #[test]
    fn test_metadata_preserved() {
        let dims = [4, 4, 4];
        let origin = [10.0, -5.5, 3.14];
        let spacing = [0.5, 0.75, 1.25];
        let vals = vec![7.0_f32; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, origin, spacing);

        let filter = MedianFilter::new(1);
        let out = filter.apply(&img).unwrap();

        // Shape.
        assert_eq!(out.shape(), dims);

        // Origin (exact equality; no computation on these values).
        let out_origin = out.origin();
        for d in 0..3 {
            assert!(
                (out_origin[d] - origin[d]).abs() < 1e-12,
                "origin[{d}]: expected {}, got {}",
                origin[d],
                out_origin[d]
            );
        }

        // Spacing.
        let out_spacing = out.spacing();
        for d in 0..3 {
            assert!(
                (out_spacing[d] - spacing[d]).abs() < 1e-12,
                "spacing[{d}]: expected {}, got {}",
                spacing[d],
                out_spacing[d]
            );
        }

        // Direction (identity → identity).
        let out_dir = out.direction();
        let in_dir = img.direction();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (out_dir[(i, j)] - in_dir[(i, j)]).abs() < 1e-12,
                    "direction[{i},{j}] mismatch"
                );
            }
        }
    }

    // ── Test 4: Identity for radius zero ──────────────────────────────────

    /// With radius = 0 the neighbourhood is a single voxel, so the output
    /// must be bit-identical to the input.
    ///
    /// **Proof**: |N(p)| = 1³ = 1, so median of {I(p)} = I(p). ∎
    #[test]
    fn test_identity_for_radius_zero() {
        let dims = [6, 6, 6];
        let n = dims[0] * dims[1] * dims[2];
        // Deterministic non-trivial pattern: voxel value = flat index.
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img = make_image(vals.clone(), dims, [0.0; 3], [1.0; 3]);

        let filter = MedianFilter::new(0);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        assert_eq!(result.len(), vals.len());
        for (i, (&expected, &actual)) in vals.iter().zip(result.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "voxel {i}: expected {expected}, got {actual}"
            );
        }
    }
}
