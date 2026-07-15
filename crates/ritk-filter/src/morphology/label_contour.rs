//! Label contour filter — boundaries between labelled regions.
//!
//! # Mathematical Specification
//!
//! Given a label image `L : ℤ³ → ℕ₀`, a voxel `p` is a **label contour voxel** if:
//! - `L(p) ≠ background_label`, AND
//! - at least one neighbour `q ∈ N(p)` satisfies `L(q) ≠ L(p)`.
//!
//! The connectivity topology `N(p)` is determined by [`Connectivity`]:
//! - [`Connectivity::Face6`] (default): 6-connected face neighbours (ITK default).
//! - [`Connectivity::Vertex26`]: 26-connected neighbours.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::LabelContourImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `FullyConnected = false`, `BackgroundValue = 0`.
//!
//! Output: a contour voxel retains its original label value;
//! non-contour and background voxels are set to `background_value` (default 0).
//!
//! # Reference
//!
//! - Malandain, G. & Bertrand, G. (1992). Fast characterization of 3D simple points.

use super::Connectivity;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Label contour filter.
///
/// Marks voxels that lie on the boundary between distinct labelled regions.
#[derive(Debug, Clone)]
pub struct LabelContourImageFilter {
    /// Neighbourhood connectivity topology (ITK default: `Face6`).
    pub connectivity: Connectivity,
    /// Label value used for background. Default 0.
    pub background_value: f32,
}

impl LabelContourImageFilter {
    /// Construct with explicit parameters.
    pub fn new(connectivity: Connectivity, background_value: f32) -> Self {
        Self {
            connectivity,
            background_value,
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

impl Default for LabelContourImageFilter {
    fn default() -> Self {
        Self::new(Connectivity::Face6, 0.0)
    }
}

/// 6-connected face neighbours.
const N6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

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

impl LabelContourImageFilter {
    /// Apply the label contour filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;

        let bg = self.background_value;
        let n26 = n26();

        let mut out = vec![bg; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let label = vals[iz * ny * nx + iy * nx + ix];
                    if (label - bg).abs() < 1e-5 {
                        continue; // background stays background
                    }
                    // A labelled voxel is a contour voxel iff an IN-BOUNDS
                    // neighbour has a different label. Out-of-bounds neighbours
                    // are skipped, NOT treated as a different label — ITK /
                    // `sitk.LabelContour` leaves a single full-label image empty
                    // and never marks the image border. (Treating OOB as a
                    // different label also broke z=1 images.)
                    let neighbour_differs = |dz: i32, dy: i32, dx: i32| -> bool {
                        let qz = iz as i32 + dz;
                        let qy = iy as i32 + dy;
                        let qx = ix as i32 + dx;
                        if qz < 0
                            || qy < 0
                            || qx < 0
                            || qz >= nz as i32
                            || qy >= ny as i32
                            || qx >= nx as i32
                        {
                            return false;
                        }
                        let nl = vals[qz as usize * ny * nx + qy as usize * nx + qx as usize];
                        (nl - label).abs() > 1e-5
                    };
                    let is_contour = match self.connectivity {
                        Connectivity::Vertex26 => n26
                            .iter()
                            .any(|&(dz, dy, dx)| neighbour_differs(dz, dy, dx)),
                        Connectivity::Face6 => {
                            N6.iter().any(|&(dz, dy, dx)| neighbour_differs(dz, dy, dx))
                        }
                    };
                    if is_contour {
                        out[iz * ny * nx + iy * nx + ix] = label;
                    }
                }
            }
        }

        Ok(rebuild(out, dims, image))
    }

    /// Apply label contour extraction to a Coeus-native image.
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
        let background = self.background_value;
        let n26 = n26();
        let slab = ny * nx;
        let connectivity = self.connectivity;
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |flat| {
            let label = values[flat];
            if (label - background).abs() < 1e-5 {
                return background;
            }
            let z = flat / slab;
            let rem = flat - z * slab;
            let y = rem / nx;
            let x = rem - y * nx;
            let differs = |dz: i32, dy: i32, dx: i32| {
                let [nz_, ny_, nx_] = [z as i32 + dz, y as i32 + dy, x as i32 + dx];
                nz_ >= 0
                    && ny_ >= 0
                    && nx_ >= 0
                    && nz_ < nz as i32
                    && ny_ < ny as i32
                    && nx_ < nx as i32
                    && (values[nz_ as usize * slab + ny_ as usize * nx + nx_ as usize] - label)
                        .abs()
                        > 1e-5
            };
            let contour = match connectivity {
                Connectivity::Vertex26 => n26.iter().any(|&(z, y, x)| differs(z, y, x)),
                Connectivity::Face6 => N6.iter().any(|&(z, y, x)| differs(z, y, x)),
            };
            if contour {
                label
            } else {
                background
            }
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_label_contour.rs"]
mod tests_label_contour;
