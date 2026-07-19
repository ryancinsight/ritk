use super::ops::{fold_native, fold_wide, project_any, project_median, project_stddev};
use super::ProjectionAxis;
use anyhow::Result;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;

// ── MaxIntensityProjectionFilter ──────────────────────────────────────────────

/// Maximum intensity projection along a chosen axis.
pub struct MaxIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MaxIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        fold_native(self.axis, image, f32::NEG_INFINITY, |a, b| {
            if b > a {
                b
            } else {
                a
            }
        })
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let [nz, ny, nx] = image.shape();
        let (vals, _) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let out = match self.axis {
            ProjectionAxis::Z => {
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(ny * nx, |idx| {
                    let y = idx / nx;
                    let x = idx % nx;
                    (0..nz).fold(f32::NEG_INFINITY, |acc, z| {
                        acc.max(vals[z * ny * nx + y * nx + x])
                    })
                })
            }
            ProjectionAxis::Y => {
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * nx, |idx| {
                    let z = idx / nx;
                    let x = idx % nx;
                    (0..ny).fold(f32::NEG_INFINITY, |acc, y| {
                        acc.max(vals[z * ny * nx + y * nx + x])
                    })
                })
            }
            ProjectionAxis::X => {
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny, |idx| {
                    let z = idx / ny;
                    let y = idx % ny;
                    (0..nx).fold(f32::NEG_INFINITY, |acc, x| {
                        acc.max(vals[z * ny * nx + y * nx + x])
                    })
                })
            }
        };
        let dims = match self.axis {
            ProjectionAxis::Z => [1, ny, nx],
            ProjectionAxis::Y => [nz, 1, nx],
            ProjectionAxis::X => [nz, ny, 1],
        };
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

// ── MinIntensityProjectionFilter ──────────────────────────────────────────────

/// Minimum intensity projection along a chosen axis.
pub struct MinIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MinIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        fold_native(
            self.axis,
            image,
            f32::INFINITY,
            |a, b| {
                if b < a {
                    b
                } else {
                    a
                }
            },
        )
    }
}

// ── MeanIntensityProjectionFilter ─────────────────────────────────────────────

/// Mean intensity projection along a chosen axis (f64 accumulation).
pub struct MeanIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MeanIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        fold_wide(self.axis, image, |sum, n| (sum / n as f64) as f32)
    }
}

// ── SumIntensityProjectionFilter ──────────────────────────────────────────────

/// Sum intensity projection along a chosen axis (f64 accumulation).
pub struct SumIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl SumIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        fold_wide(self.axis, image, |sum, _n| sum as f32)
    }
}

// ── StdDevIntensityProjectionFilter ───────────────────────────────────────────

/// Population standard-deviation projection along a chosen axis (f64 accumulation).
pub struct StdDevIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl StdDevIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        project_stddev(self.axis, image)
    }
}

// ── MedianIntensityProjectionFilter ───────────────────────────────────────────

/// Median intensity projection along a chosen axis.
pub struct MedianIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MedianIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        project_median(self.axis, image)
    }
}

// ── BinaryProjectionFilter ────────────────────────────────────────────────────

/// Binary projection along a chosen axis: a result pixel is `foreground` if
/// **any** voxel along the collapsed axis equals `foreground`, else `background`.
pub struct BinaryProjectionFilter {
    axis: ProjectionAxis,
    foreground: f32,
    background: f32,
}

impl BinaryProjectionFilter {
    pub fn new(axis: ProjectionAxis, foreground: f32, background: f32) -> Self {
        Self {
            axis,
            foreground,
            background,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let fg = self.foreground;
        project_any(
            self.axis,
            image,
            self.foreground,
            self.background,
            move |v| v == fg,
        )
    }
}

// ── BinaryThresholdProjectionFilter ───────────────────────────────────────────

/// Binary-threshold projection: a result pixel is `foreground` if **any** voxel
/// along the collapsed axis is `>= threshold`, else `background`.
pub struct BinaryThresholdProjectionFilter {
    axis: ProjectionAxis,
    threshold: f32,
    foreground: f32,
    background: f32,
}

impl BinaryThresholdProjectionFilter {
    pub fn new(axis: ProjectionAxis, threshold: f32, foreground: f32, background: f32) -> Self {
        Self {
            axis,
            threshold,
            foreground,
            background,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let thr = self.threshold;
        project_any(
            self.axis,
            image,
            self.foreground,
            self.background,
            move |v| v >= thr,
        )
    }
}
