//! Box sigma (sliding standard-deviation) filter (`itk::BoxSigmaImageFilter`).
//!
//! # Mathematical Specification
//!
//! For per-axis radii `[rz, ry, rx]`, each voxel is replaced by the **sample**
//! standard deviation (Bessel-corrected, divisor `n − 1`) over the axis-aligned
//! `(2r+1)` window clipped to the image bounds:
//!
//! ```text
//! W   = ([z−rz, z+rz] × [y−ry, y+ry] × [x−rx, x+rx]) ∩ image,  n = |W|
//! out = sqrt( (Σ_{k∈W} I(k)² − (Σ_{k∈W} I(k))² / n) / (n − 1) )
//! ```
//!
//! Windows with `n ≤ 1` yield `0`. Matches `itk::BoxSigmaImageFilter` /
//! `sitk.BoxSigma`, which uses the sample (not population) divisor — pinned by a
//! probe: `[10,20,30,40,50]` r=1 → interior `[20,30,40]` gives `10` (sample),
//! not `8.165` (population); the clipped boundary window `[10,20]` gives
//! `7.071`. Shares the clipped-window/shrink-boundary convention with
//! [`super::box_mean::BoxMeanImageFilter`].

use coeus_core::{CpuAddressableStorage, Scalar};
use ritk_image::tensor::Backend;
use ritk_image::{Image, RowWalker, VoxelRegion};
use ritk_tensor_ops::rebuild;

/// Box sigma filter — clipped-window sample standard deviation
/// (ITK `BoxSigmaImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct BoxSigmaImageFilter {
    /// Per-axis radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
}

impl BoxSigmaImageFilter {
    /// Construct with the given per-axis radii.
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Apply the box sigma to a 3-D image.
    ///
    /// # Panics
    ///
    /// Panics when the image's backing tensor rank does not match its type-level
    /// rank, which `Image`'s constructors already preclude.
    pub fn apply<B>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3>
    where
        B: Backend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let region = image
            .region()
            .expect("invariant: Image::new validates tensor rank equals D");
        rebuild(self.sigma_over(&region), image.shape(), image)
    }

    /// Coeus-native counterpart to the legacy application method.
    ///
    /// # Errors
    ///
    /// Returns an error when the image cannot be viewed as a region, or when the
    /// result cannot be rebuilt into an image.
    pub fn apply_native<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let region = image.region()?;
        let out = self.sigma_over(&region);
        crate::native_support::rebuild_image(out, image.shape(), image, backend)
    }

    /// Clipped-window sample standard deviation over a borrowed region.
    ///
    /// The single kernel behind both entry points. It reads the input through
    /// the region seam rather than a flat host copy: the window per output voxel
    /// is a [`VoxelRegion::clipped_window`] (fixed-size arrays, no allocation),
    /// and its values are accumulated over the lent innermost rows, which are
    /// direct borrows of the image buffer whenever the inner axis is unit-stride
    /// — every ordinary volume. The whole traversal allocates only the output.
    fn sigma_over<T>(&self, region: &VoxelRegion<'_, T, 3>) -> Vec<f32>
    where
        T: Scalar + Copy + Into<f64>,
    {
        let shape = region.shape();
        let [_, ny, nx] = shape;
        let radius = self.radius;
        let plane = ny * nx;

        // Per-voxel independent (each output reads only its clipped window), so
        // the grid fans out across threads; the result is bitwise identical to a
        // serial run.
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(region.len(), move |flat| {
            let z = flat / plane;
            let rem = flat % plane;
            let centre = [z, rem / nx, rem % nx];
            let window = region
                .clipped_window(centre, radius)
                .expect("invariant: centre derives from an in-range flat index");

            let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
            let mut rows = window.rows();
            while let Some(row) = rows.next_row() {
                for &value in row {
                    let v: f64 = value.into();
                    sum += v;
                    sumsq += v * v;
                }
            }

            let n = window.len() as f64;
            if n > 1.0 {
                let var = (sumsq - sum * sum / n) / (n - 1.0);
                var.max(0.0).sqrt() as f32
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
#[path = "tests_box_sigma.rs"]
mod tests_box_sigma;
