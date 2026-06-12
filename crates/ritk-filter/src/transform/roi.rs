//! Region-of-interest (crop) filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given an image I with shape `[Nz, Ny, Nx]`, origin `O`, spacing `S`, and
//! direction matrix `D`, the ROI filter extracts the sub-volume:
//!
//! `out[iz][iy][ix] = I[start_z + iz][start_y + iy][start_x + ix]`
//!
//! for `iz ∈ [0, size_z)`, `iy ∈ [0, size_y)`, `ix ∈ [0, size_x)`.
//!
//! ## Origin Update
//!
//! The physical origin of the cropped image is the physical coordinate of
//! voxel `(start_z, start_y, start_x)` in the input image:
//!
//! `new_origin[k] = O[k]
//!     + start_x * S[2] * D.col(2)[k]
//!     + start_y * S[1] * D.col(1)[k]
//!     + start_z * S[0] * D.col(0)[k]`
//!
//! where RITK uses ZYX tensor ordering with direction columns:
//! - `D.col(0)` = Z-axis direction
//! - `D.col(1)` = Y-axis direction
//! - `D.col(2)` = X-axis direction
//!
//! ## Invariants
//!
//! - `start_k + size_k ≤ N_k` for k ∈ {z, y, x} (validated at runtime).
//! - `size_k ≥ 1` for all k.
//! - Output shape = `[size_z, size_y, size_x]`.
//! - Spacing and direction are preserved exactly.
//! - Origin is updated to the physical position of the first cropped voxel.
//!
//! # ITK Parity
//!
//! `itk::RegionOfInterestImageFilter` with `SetRegionOfInterest(region)`.

use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_origin};
use ritk_image::Image;
use ritk_spatial::Point;
use burn::tensor::backend::Backend;

/// Extract a 3-D sub-volume (region of interest) from an image.
///
/// All indices are in ZYX order matching RITK's tensor convention.
#[derive(Debug, Clone)]
pub struct RegionOfInterestImageFilter {
    /// Starting voxel index in Z (slowest axis).
    pub start_z: usize,
    /// Starting voxel index in Y.
    pub start_y: usize,
    /// Starting voxel index in X (fastest axis).
    pub start_x: usize,
    /// Number of voxels to extract in Z.
    pub size_z: usize,
    /// Number of voxels to extract in Y.
    pub size_y: usize,
    /// Number of voxels to extract in X.
    pub size_x: usize,
}

impl RegionOfInterestImageFilter {
    /// Create a new ROI filter.
    ///
    /// `start` = `[start_z, start_y, start_x]`  
    /// `size`  = `[size_z, size_y, size_x]`
    pub fn new(start: [usize; 3], size: [usize; 3]) -> Self {
        Self {
            start_z: start[0],
            start_y: start[1],
            start_x: start[2],
            size_z: size[0],
            size_y: size[1],
            size_x: size[2],
        }
    }

    /// Apply the ROI filter to a 3-D image.
    ///
    /// Returns `Err` if the requested region exceeds the image bounds or if
    /// any size dimension is zero.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();

        // Validate bounds
        if self.size_z == 0 || self.size_y == 0 || self.size_x == 0 {
            anyhow::bail!("ROI size must be ≥ 1 in all dimensions");
        }
        if self.start_z + self.size_z > nz {
            anyhow::bail!(
                "ROI Z range [{}..{}) exceeds image depth {}",
                self.start_z,
                self.start_z + self.size_z,
                nz
            );
        }
        if self.start_y + self.size_y > ny {
            anyhow::bail!(
                "ROI Y range [{}..{}) exceeds image height {}",
                self.start_y,
                self.start_y + self.size_y,
                ny
            );
        }
        if self.start_x + self.size_x > nx {
            anyhow::bail!(
                "ROI X range [{}..{}) exceeds image width {}",
                self.start_x,
                self.start_x + self.size_x,
                nx
            );
        }

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

        // Extract sub-volume
        let sz = self.size_z;
        let sy = self.size_y;
        let sx = self.size_x;
        let mut out = vec![0.0f32; sz * sy * sx];
        for iz in 0..sz {
            for iy in 0..sy {
                for ix in 0..sx {
                    let src = (self.start_z + iz) * ny * nx
                        + (self.start_y + iy) * nx
                        + (self.start_x + ix);
                    let dst = iz * sy * sx + iy * sx + ix;
                    out[dst] = vals[src];
                }
            }
        }

        // Update origin: physical position of voxel (start_z, start_y, start_x)
        // new_origin[k] = old_origin[k]
        //   + start_z * spacing[0] * direction.col(0)[k]
        //   + start_y * spacing[1] * direction.col(1)[k]
        //   + start_x * spacing[2] * direction.col(2)[k]
        let old_origin = image.origin();
        let spacing = image.spacing();
        let dir = image.direction();
        let mut new_coords = [0.0f64; 3];
        for (k, coord) in new_coords.iter_mut().enumerate() {
            *coord = old_origin.0.coords[k]
                + self.start_z as f64 * spacing[0] * dir.0[(k, 0)]
                + self.start_y as f64 * spacing[1] * dir.0[(k, 1)]
                + self.start_x as f64 * spacing[2] * dir.0[(k, 2)];
        }
        let new_origin = Point::new(new_coords);

        Ok(rebuild_with_origin(out, [sz, sy, sx], new_origin, image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        use burn::tensor::{Shape, Tensor, TensorData};
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    #[test]
    fn roi_full_image_is_identity() {
        // ROI spanning the whole volume = identical copy
        let vals: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        let img = make_image(vals.clone(), [3, 3, 3]);
        let out = RegionOfInterestImageFilter::new([0, 0, 0], [3, 3, 3])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [3, 3, 3]);
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert_eq!(a, b, "voxel {}: identity violation", i);
        }
    }

    #[test]
    fn roi_extracts_correct_sub_volume() {
        // 3×3×3 with values 1..27; extract [1..3, 1..3, 1..3] (2×2×2)
        let vals: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        let img = make_image(vals, [3, 3, 3]);
        let out = RegionOfInterestImageFilter::new([1, 1, 1], [2, 2, 2])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        let v = voxels(&out);
        // Voxel (1,1,1) in 3×3×3 = index 1*9 + 1*3 + 1 = 13 → value 14
        assert_eq!(v[0], 14.0, "first voxel of sub-volume mismatch");
        // Voxel (1,1,2) = 1*9 + 1*3 + 2 = 14 → value 15
        assert_eq!(v[1], 15.0);
        // Voxel (2,2,2) = 2*9 + 2*3 + 2 = 26 → value 27
        assert_eq!(v[7], 27.0, "last voxel of sub-volume mismatch");
    }

    #[test]
    fn roi_single_voxel_extract() {
        let vals: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let img = make_image(vals, [2, 2, 2]);
        // Extract voxel (1,1,1) — index 7, value 8
        let out = RegionOfInterestImageFilter::new([1, 1, 1], [1, 1, 1])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [1, 1, 1]);
        let v = voxels(&out);
        assert_eq!(v[0], 8.0, "expected voxel (1,1,1) = 8");
    }

    #[test]
    fn roi_updates_origin_with_identity_direction() {
        // With identity direction and spacing=[2,3,4], start=[1,1,1]:
        // new_origin[0] = 0 + 1*2*1 = 2
        // new_origin[1] = 0 + 1*3*1 = 3
        // new_origin[2] = 0 + 1*4*1 = 4
        use burn::tensor::{Shape, Tensor, TensorData};
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![0.0f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            tensor,
            Point::new([0.0f64, 0.0, 0.0]),
            Spacing::new([2.0f64, 3.0, 4.0]),
            Direction::identity(),
        );
        let out = RegionOfInterestImageFilter::new([1, 1, 1], [2, 2, 2])
            .apply(&img)
            .unwrap();
        let o = out.origin();
        assert!(
            (o.0.coords[0] - 2.0).abs() < 1e-9,
            "origin[0] expected 2.0 got {}",
            o.0.coords[0]
        );
        assert!(
            (o.0.coords[1] - 3.0).abs() < 1e-9,
            "origin[1] expected 3.0 got {}",
            o.0.coords[1]
        );
        assert!(
            (o.0.coords[2] - 4.0).abs() < 1e-9,
            "origin[2] expected 4.0 got {}",
            o.0.coords[2]
        );
    }

    #[test]
    fn roi_out_of_bounds_returns_error() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        // start=[0,0,0], size=[3,2,2] → Z range [0..3) exceeds depth 2
        let result = RegionOfInterestImageFilter::new([0, 0, 0], [3, 2, 2]).apply(&img);
        assert!(result.is_err(), "out-of-bounds ROI must return Err");
    }
}
