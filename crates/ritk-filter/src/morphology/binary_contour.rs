//! Binary contour filter — foreground border voxels of binary objects.
//!
//! # Mathematical Specification
//!
//! Given a binary image `I ∈ {0, fg}`, a voxel `p` is a **border voxel** if:
//! - `I(p) = fg` (is foreground), AND
//! - at least one neighbour `q ∈ N(p)` satisfies `I(q) ≠ fg`.
//!
//! The connectivity topology `N(p)` is determined by [`Connectivity`]:
//! - [`Connectivity::Face6`] (default): 6-connected — 6 face-neighbours in ℤ³ (ITK default).
//! - [`Connectivity::Vertex26`]: 26-connected — all 26 neighbours within the unit cube.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::BinaryContourImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `FullyConnected = false`, `ForegroundValue = 1`, `BackgroundValue = 0`.
//!
//! Output: foreground border voxels set to `foreground_value`; all others 0.
//!
//! # Reference
//!
//! - Malandain, G. & Bertrand, G. (1992). Fast characterization of 3D simple points.
//!   *ICPR 1992*.

use super::types::ForegroundValue;
use super::Connectivity;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

/// Binary contour filter.
///
/// Marks only the border (surface) voxels of foreground objects.
/// Interior foreground voxels (fully surrounded by foreground) are set to 0.
#[derive(Debug, Clone)]
pub struct BinaryContourImageFilter {
    /// Neighbourhood connectivity topology (ITK default: `Face6`).
    pub connectivity: Connectivity,
    /// Foreground intensity value. Default 1.0.
    pub foreground_value: ForegroundValue,
}

impl BinaryContourImageFilter {
    /// Construct with explicit parameters.
    pub fn new(connectivity: Connectivity, foreground_value: impl Into<ForegroundValue>) -> Self {
        Self {
            connectivity,
            foreground_value: foreground_value.into(),
        }
    }

    /// Set connectivity.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Backward-compatible accessor for the former `fully_connected: bool` field.
    pub fn fully_connected(&self) -> bool {
        self.connectivity.fully_connected()
    }
}

impl Default for BinaryContourImageFilter {
    fn default() -> Self {
        Self::new(Connectivity::Face6, ForegroundValue::ONE)
    }
}

/// 6-connected face neighbours (±z, ±y, ±x).
const N6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// 26-connected neighbours (all offsets in {-1,0,1}³ except (0,0,0)).
fn n26() -> Vec<(i32, i32, i32)> {
    let mut v = Vec::with_capacity(26);
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dz != 0 || dy != 0 || dx != 0 {
                    v.push((dz, dy, dx));
                }
            }
        }
    }
    v
}

impl BinaryContourImageFilter {
    /// Apply the binary contour filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = self.contour_values(&vals, dims);
        let device = image.data().device();
        let shape = Shape::new(dims);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply binary contour extraction to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            self.contour_values(image.data_slice()?, image.shape()),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn contour_values(&self, values: &[f32], [nz, ny, nx]: [usize; 3]) -> Vec<f32> {
        let fg = f32::from(self.foreground_value);
        let n26 = n26();
        let slab = ny * nx;
        let connectivity = self.connectivity;
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |flat| {
            if (values[flat] - fg).abs() > 1e-5 {
                return 0.0;
            }
            let iz = flat / slab;
            let rem = flat - iz * slab;
            let iy = rem / nx;
            let ix = rem - iy * nx;
            let is_background = |dz: i32, dy: i32, dx: i32| {
                let [z, y, x] = [iz as i32 + dz, iy as i32 + dy, ix as i32 + dx];
                z >= 0
                    && y >= 0
                    && x >= 0
                    && z < nz as i32
                    && y < ny as i32
                    && x < nx as i32
                    && (values[z as usize * slab + y as usize * nx + x as usize] - fg).abs() > 1e-5
            };
            let border = match connectivity {
                Connectivity::Vertex26 => n26.iter().any(|&(z, y, x)| is_background(z, y, x)),
                Connectivity::Face6 => N6.iter().any(|&(z, y, x)| is_background(z, y, x)),
            };
            if border {
                fg
            } else {
                0.0
            }
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_contour.rs"]
mod tests_binary_contour;
