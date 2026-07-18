//! Region-of-interest (crop) filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given an image I with shape `[Nz, Ny, Nx]`, origin `O`, spacing `S`, and
//! direction matrix `D`, the ROI filter extracts the sub-volume:
//!
//! `out[iz][iy][ix] = I[start_z + iz][start_y + iy][start_x + ix]`
//!
//! for `iz âˆˆ [0, size_z)`, `iy âˆˆ [0, size_y)`, `ix âˆˆ [0, size_x)`.
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
//! - `start_k + size_k â‰¤ N_k` for k âˆˆ {z, y, x} (validated at runtime).
//! - `size_k â‰¥ 1` for all k.
//! - Output shape = `[size_z, size_y, size_x]`.
//! - Spacing and direction are preserved exactly.
//! - Origin is updated to the physical position of the first cropped voxel.
//!
//! # ITK Parity
//!
//! `itk::RegionOfInterestImageFilter` with `SetRegionOfInterest(region)`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_origin};

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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        self.validate_region(image.shape())?;

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let out = self.crop_values(vals, image.shape());

        // Update origin: physical position of voxel (start_z, start_y, start_x)
        // new_origin[k] = old_origin[k]
        //   + start_z * spacing[0] * direction.col(0)[k]
        //   + start_y * spacing[1] * direction.col(1)[k]
        //   + start_x * spacing[2] * direction.col(2)[k]
        let new_origin = self.cropped_origin(image.origin(), image.spacing(), image.direction());

        Ok(rebuild_with_origin(
            out,
            [self.size_z, self.size_y, self.size_x],
            new_origin,
            image,
        ))
    }

    /// Apply the ROI crop to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.validate_region(image.shape())?;
        let spacing = image.spacing();
        let direction = image.direction();
        let origin = image.origin();
        let values = image.data_slice()?;
        let new_origin = self.cropped_origin(origin, spacing, direction);
        let output = self.crop_values(values, image.shape());
        ritk_image::native::Image::from_flat_on(
            output,
            [self.size_z, self.size_y, self.size_x],
            new_origin,
            *spacing,
            *direction,
            backend,
        )
    }

    fn validate_region(&self, [nz, ny, nx]: [usize; 3]) -> anyhow::Result<()> {
        if self.size_z == 0 || self.size_y == 0 || self.size_x == 0 {
            anyhow::bail!("ROI size must be >= 1 in all dimensions");
        }
        if self.start_z + self.size_z > nz
            || self.start_y + self.size_y > ny
            || self.start_x + self.size_x > nx
        {
            anyhow::bail!("ROI range exceeds image bounds");
        }
        Ok(())
    }

    fn crop_values(&self, values: &[f32], [_, ny, nx]: [usize; 3]) -> Vec<f32> {
        let mut output = vec![0.0; self.size_z * self.size_y * self.size_x];
        for z in 0..self.size_z {
            for y in 0..self.size_y {
                for x in 0..self.size_x {
                    output[z * self.size_y * self.size_x + y * self.size_x + x] = values
                        [(self.start_z + z) * ny * nx + (self.start_y + y) * nx + self.start_x + x];
                }
            }
        }
        output
    }

    fn cropped_origin(
        &self,
        origin: &Point<3>,
        spacing: &Spacing<3>,
        direction: &Direction<3>,
    ) -> Point<3> {
        Point::new(std::array::from_fn(|axis| {
            origin[axis]
                + self.start_z as f64 * spacing[0] * direction[(axis, 0)]
                + self.start_y as f64 * spacing[1] * direction[(axis, 1)]
                + self.start_x as f64 * spacing[2] * direction[(axis, 2)]
        }))
    }
}

#[cfg(test)]
#[path = "tests_roi.rs"]
mod tests_roi;
