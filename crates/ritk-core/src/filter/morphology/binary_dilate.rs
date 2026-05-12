//! Binary dilation filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary dilation with a flat cubic structuring element B of half-width r:
//!
//!   (D_B f)(x) = fg  iff  ∃ b ∈ B: f(x − b) = fg
//!             = bg  otherwise
//!
//! Equivalently: a voxel x is foreground in the output iff any voxel in its
//! `(2r+1)³` cubic neighbourhood is foreground in the input.
//!
//! # Boundary Handling
//!
//! Out-of-bounds positions are treated as background — they do not contribute
//! foreground.  This is equivalent to `itk::BinaryDilateImageFilter` with
//! `BoundaryToForeground = false` (the ITK default).
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryDilateImageFilter` with:
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

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary dilation filter for 3-D images.
///
/// Grows foreground regions by one layer of `radius` voxels.  Each voxel is
/// foreground in the output iff at least one voxel in its `(2r+1)³` cubic
/// neighbourhood is foreground in the input.
///
/// Out-of-bounds positions are treated as background, so the foreground
/// region cannot grow beyond the image boundary.
#[derive(Debug, Clone)]
pub struct BinaryDilateFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: f32,
}

impl BinaryDilateFilter {
    /// Create a binary dilation filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: 1.0,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: f32) -> Self {
        self.foreground_value = v;
        self
    }

    /// Apply binary dilation to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("BinaryDilateFilter requires f32 data: {:?}", e))?;

        let result = dilate_binary_3d(vals, dims, self.radius, self.foreground_value);

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

impl Default for BinaryDilateFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Binary dilation on a flat Z×Y×X volume.
///
/// # Invariants
///
/// - Output length = `nz × ny × nx`.
/// - Output[i] ∈ {foreground_value, 0.0}.
/// - Output[i] = foreground_value iff any (2r+1)³ in-bounds neighbour = fg.
pub(crate) fn dilate_binary_3d(data: &[f32], dims: [usize; 3], radius: usize, fg: f32) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut any_fg = false;
                'outer: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = iz as isize + dz;
                            let yy = iy as isize + dy;
                            let xx = ix as isize + dx;
                            if zz < 0
                                || yy < 0
                                || xx < 0
                                || zz >= nz as isize
                                || yy >= ny as isize
                                || xx >= nx as isize
                            {
                                continue;
                            }
                            let idx = zz as usize * ny * nx + yy as usize * nx + xx as usize;
                            if data[idx] == fg {
                                any_fg = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if any_fg {
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
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
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

    /// T1: Radius-0 dilation is identity.
    #[test]
    fn radius_zero_is_identity() {
        let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let out = BinaryDilateFilter::new(0).apply(&img).unwrap();
        assert_eq!(flat(&out), vals);
    }

    /// T2: Single foreground voxel dilates to (2r+1)³ cube.
    ///
    /// 1×1×5 image with fg at centre (index 2), r=1:
    /// Expected output: [0, fg, fg, fg, 0] — centre ± 1.
    #[test]
    fn single_voxel_dilates_to_cube() {
        let img = make_image(vec![0.0, 0.0, 1.0, 0.0, 0.0], [1, 1, 5]);
        let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
        assert_eq!(flat(&out), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    /// T3: r=1 on a 1×1×5 image, fg at index 0 — cannot dilate left (border).
    /// Expected: [fg, fg, 0, 0, 0].
    #[test]
    fn border_dilation_bounded_by_image_edge() {
        let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 5]);
        let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
        assert_eq!(flat(&out), vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    /// T4: All-background image → all background after dilation.
    #[test]
    fn all_background_unchanged() {
        let img = make_image(vec![0.0; 8], [2, 2, 2]);
        let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
        assert!(flat(&out).iter().all(|&v| v == 0.0));
    }

    /// T5: All-foreground image → all foreground after dilation.
    #[test]
    fn all_foreground_unchanged() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
        assert!(flat(&out).iter().all(|&v| v == 1.0));
    }

    /// T6: Custom foreground value 255.0.
    #[test]
    fn custom_foreground_value() {
        let img = make_image(vec![0.0, 0.0, 255.0, 0.0, 0.0], [1, 1, 5]);
        let out = BinaryDilateFilter::new(1)
            .with_foreground(255.0)
            .apply(&img)
            .unwrap();
        assert_eq!(flat(&out), vec![0.0, 255.0, 255.0, 255.0, 0.0]);
    }

    /// T7: Dilation produces a known analytically correct output.
    ///
    /// Input `f = [0, 0, 1, 0, 0]` in 1×1×5.  With r=1, each voxel is fg if
    /// EXISTS an in-bounds X-neighbour that is fg (Z/Y neighbours are OOB at
    /// nz=ny=1, but dilation only requires EXISTS — OOB does not contribute fg).
    ///
    /// Voxel (0,0,0): in-bounds X-neighbours = {(0,0,0)=0,(0,0,1)=0} → no fg → bg.
    /// Voxel (0,0,1): in-bounds X-neighbour (0,0,2) = 1 → fg.
    /// Voxel (0,0,2): self = 1 → fg.
    /// Voxel (0,0,3): in-bounds X-neighbour (0,0,2) = 1 → fg.
    /// Voxel (0,0,4): in-bounds X-neighbours = {(0,0,3)=0,(0,0,4)=0} → no fg → bg.
    ///
    /// Expected: [0, 1, 1, 1, 0].
    #[test]
    fn dilation_known_output() {
        let f: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let img = make_image(f, [1, 1, 5]);
        let out = BinaryDilateFilter::new(1).apply(&img).unwrap();
        assert_eq!(flat(&out), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    /// T8: Spatial metadata is preserved unchanged.
    #[test]
    fn spatial_metadata_preserved() {
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 1.0]);
        let direction = Direction::identity();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
            &device,
        );
        let img = Image::new(t, origin, spacing, direction);
        let out = BinaryDilateFilter::new(0).apply(&img).unwrap();
        assert_eq!(*out.origin(), origin);
        assert_eq!(*out.spacing(), spacing);
        assert_eq!(*out.direction(), direction);
    }
}
