//! Binary erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary erosion with a flat cubic structuring element B of half-width r:
//!
//!   (E_B f)(x) = fg  iff  ∀ b ∈ B: f(x + b) = fg
//!             = bg  otherwise
//!
//! where B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }.
//!
//! # Boundary Handling
//!
//! Out-of-bounds neighbours are treated as background (`bg`).  This causes
//! erosion to remove the foreground layer at the image border — consistent
//! with `itk::BinaryErodeImageFilter` when `BoundaryToForeground = false`
//! (the ITK default).
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryErodeImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetBackgroundValue(0.0)`
//! - `SetBoundaryToForeground(false)` (default)
//! - Flat ball structuring element of radius r.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) where N is the total voxel count.
//!
//! # References
//!
//! - Haralick, R.M., Sternberg, S.R., & Zhuang, X. (1987). Image analysis
//!   using mathematical morphology. *IEEE TPAMI*, 9(4), 532–550.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::types::ForegroundValue;
use ritk_tensor_ops::extract_vec;
use ritk_image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary erosion filter for 3-D images.
///
/// Shrinks foreground regions by eroding their boundaries.  Each voxel is
/// foreground in the output iff every voxel in its `(2r+1)³` cubic
/// neighbourhood is foreground in the input.
///
/// Out-of-bounds neighbours are treated as background, so foreground regions
/// touching the image border are eroded to background (ITK default behaviour).
#[derive(Debug, Clone)]
pub struct BinaryErodeFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryErodeFilter {
    /// Create a binary erosion filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary erosion to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let result = erode_binary_3d(&vals, dims, self.radius, self.foreground_value);

        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for BinaryErodeFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Binary erosion on a flat Z×Y×X volume.
///
/// # Invariants
///
/// - Output length = `nz × ny × nx`.
/// - Output[i] ∈ {foreground_value, 0.0}.
/// - Output[i] = foreground_value iff all (2r+1)³ neighbours (clamped-background) = fg.
pub(crate) fn erode_binary_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    fg: ForegroundValue,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let fg: f32 = fg.into();
    let n = nz * ny * nx;
    let mut output = vec![0.0_f32; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut all_fg = true;
                'outer: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = iz as isize + dz;
                            let yy = iy as isize + dy;
                            let xx = ix as isize + dx;
                            // Out-of-bounds → background
                            if zz < 0
                                || yy < 0
                                || xx < 0
                                || zz >= nz as isize
                                || yy >= ny as isize
                                || xx >= nx as isize
                            {
                                all_fg = false;
                                break 'outer;
                            }
                            let idx = zz as usize * ny * nx + yy as usize * nx + xx as usize;
                            if data[idx] != fg {
                                all_fg = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if all_fg {
                    output[iz * ny * nx + iy * nx + ix] = fg;
                }
            }
        }
    }
    output
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
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn flat(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// T1: Radius-0 erosion is identity (single-voxel SE).
    #[test]
    fn radius_zero_is_identity() {
        let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let out = BinaryErodeFilter::new(0).apply(&img).unwrap();
        assert_eq!(flat(&out), vals);
    }

    /// T2: All-foreground 3×3×3 image with r=1 → only centre voxel (1,1,1) survives.
    ///
    /// For r=1, a voxel survives erosion iff ALL 27 SE positions are in-bounds and fg.
    /// Only (1,1,1) has all neighbours within [0,2]³ (nz=ny=nx=3).
    /// All border/edge/corner voxels have at least one OOB SE position → eroded to bg.
    ///
    /// Flat index of (1,1,1) in 3×3×3: 1*9 + 1*3 + 1 = 13.
    #[test]
    fn border_voxels_eroded_to_background() {
        let img = make_image(vec![1.0; 27], [3, 3, 3]);
        let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
        let result = flat(&out);
        // Only centre voxel (flat index 13) survives.
        assert_eq!(result[13], 1.0, "centre voxel must survive erosion");
        for (i, &v) in result.iter().enumerate() {
            if i != 13 {
                assert_eq!(v, 0.0, "border/edge voxel {i} must be eroded");
            }
        }
    }

    /// T3: Background pixel surrounded by foreground is NOT changed to foreground
    ///     (erosion only removes; it cannot add foreground).
    #[test]
    fn background_remains_background() {
        // Image: [fg, bg, fg] — bg is isolated, not eroded from fg
        let img = make_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
        let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
        // Left fg eroded (out-of-bounds left), centre bg stays 0, right fg eroded.
        assert_eq!(flat(&out), vec![0.0, 0.0, 0.0]);
    }

    /// T4: 3×3×5 all-foreground, r=1 → strips one border layer from all 6 faces.
    ///
    /// Surviving voxels: iz=1, iy=1, ix ∈ {1,2,3}.
    /// Flat indices (nz=3, ny=3, nx=5): iz*15 + iy*5 + ix = 21, 22, 23.
    #[test]
    fn erosion_strips_one_border_layer_r1() {
        let img = make_image(vec![1.0; 45], [3, 3, 5]);
        let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
        let result = flat(&out);
        let mut expected = vec![0.0_f32; 45];
        expected[21] = 1.0; // (1,1,1)
        expected[22] = 1.0; // (1,1,2)
        expected[23] = 1.0; // (1,1,3)
        assert_eq!(result, expected);
    }

    /// T5: 5×5×7 all-foreground, r=2 → strips two border layers from all faces.
    ///
    /// Surviving voxels: iz=2, iy=2, ix ∈ {2,3,4}.
    /// Flat indices (nz=5, ny=5, nx=7): iz*35 + iy*7 + ix = 86, 87, 88.
    #[test]
    fn erosion_strips_two_border_layers_r2() {
        let img = make_image(vec![1.0; 175], [5, 5, 7]);
        let out = BinaryErodeFilter::new(2).apply(&img).unwrap();
        let result = flat(&out);
        let mut expected = vec![0.0_f32; 175];
        expected[86] = 1.0; // (2,2,2)
        expected[87] = 1.0; // (2,2,3)
        expected[88] = 1.0; // (2,2,4)
        assert_eq!(result, expected);
    }

    /// T6: Custom foreground value 255.0 — 3×3×5 volume, same geometry as T4.
    ///
    /// Flat indices 21, 22, 23 survive with value 255.0.
    #[test]
    fn custom_foreground_value() {
        let img = make_image(vec![255.0; 45], [3, 3, 5]);
        let out = BinaryErodeFilter::new(1)
            .with_foreground(255.0)
            .apply(&img)
            .unwrap();
        let result = flat(&out);
        let mut expected = vec![0.0_f32; 45];
        expected[21] = 255.0;
        expected[22] = 255.0;
        expected[23] = 255.0;
        assert_eq!(result, expected);
    }

    /// T7: Spatial metadata is preserved unchanged.
    #[test]
    fn spatial_metadata_preserved() {
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let origin = Point::new([3.0, 2.0, 1.0]);
        let spacing = Spacing::new([0.5, 0.5, 1.0]);
        let direction = Direction::identity();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
            &device,
        );
        let img = Image::new(t, origin, spacing, direction);
        let out = BinaryErodeFilter::new(0).apply(&img).unwrap();
        assert_eq!(*out.origin(), origin);
        assert_eq!(*out.spacing(), spacing);
        assert_eq!(*out.direction(), direction);
    }
}
