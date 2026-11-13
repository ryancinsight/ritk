//! Edit ranges of filter parameters.
//!
//! Each constant is the closed interval and edit increment the viewer offers
//! for one [`FilterKind`](super::FilterKind) parameter, so a filter panel
//! reads its slider bounds from here and clamps entered values before a
//! filter runs. A parameter without a constant accepts any value of its
//! field type (seed indices, intensity bounds, background values).

#[cfg(test)]
mod tests;

/// Closed interval `[min, max]` and edit increment of one filter parameter.
///
/// Every range in this module satisfies `min < max` and
/// `0 < step <= max - min`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterRange<T> {
    min: T,
    max: T,
    step: T,
}

impl<T> ParameterRange<T> {
    const fn new(min: T, max: T, step: T) -> Self {
        Self { min, max, step }
    }
}

impl<T: Copy + PartialOrd> ParameterRange<T> {
    /// Smallest accepted value.
    #[must_use]
    pub fn min(self) -> T {
        self.min
    }

    /// Largest accepted value.
    #[must_use]
    pub fn max(self) -> T {
        self.max
    }

    /// Increment between adjacent values offered for editing.
    #[must_use]
    pub fn step(self) -> T {
        self.step
    }

    /// Whether `value` lies in `[min, max]`; an unordered value (NaN) does not.
    #[must_use]
    pub fn contains(self, value: T) -> bool {
        value >= self.min && value <= self.max
    }

    /// `value` limited to `[min, max]`.
    ///
    /// A value below the range, or an unordered value (NaN), becomes `min`;
    /// a value above it becomes `max`.
    ///
    /// ```
    /// use ritk_snap::filter::parameter_range::GAUSSIAN_SIGMA_MM;
    ///
    /// assert_eq!(GAUSSIAN_SIGMA_MM.clamp(35.0), 20.0);
    /// assert_eq!(GAUSSIAN_SIGMA_MM.clamp(f32::NAN), 0.1);
    /// ```
    #[must_use]
    pub fn clamp(self, value: T) -> T {
        if value > self.max {
            self.max
        } else if value >= self.min {
            value
        } else {
            self.min
        }
    }
}

/// Gaussian smoothing σ in millimetres.
pub const GAUSSIAN_SIGMA_MM: ParameterRange<f32> = ParameterRange::new(0.1, 20.0, 0.1);

/// Median filter radius in voxels.
pub const MEDIAN_RADIUS: ParameterRange<usize> = ParameterRange::new(0, 10, 1);

/// CLAHE contextual tiles along each in-plane axis.
pub const CLAHE_TILES_PER_AXIS: ParameterRange<usize> = ParameterRange::new(1, 32, 1);

/// CLAHE histogram clip limit.
pub const CLAHE_CLIP_LIMIT: ParameterRange<f32> = ParameterRange::new(1.0, 200.0, 1.0);

/// Histogram-equalization bin count.
pub const HIST_EQ_BINS: ParameterRange<usize> = ParameterRange::new(2, 1024, 1);

/// Unsharp-mask blur σ in millimetres.
pub const UNSHARP_SIGMA_MM: ParameterRange<f32> = ParameterRange::new(0.1, 10.0, 0.1);

/// Unsharp-mask gain applied to the detail layer.
pub const UNSHARP_AMOUNT: ParameterRange<f32> = ParameterRange::new(0.0, 5.0, 0.05);

/// Unsharp-mask detail magnitude below which no sharpening applies.
pub const UNSHARP_THRESHOLD: ParameterRange<f32> = ParameterRange::new(0.0, 100.0, 0.5);

/// Gradient anisotropic diffusion iteration count.
pub const ANISOTROPIC_DIFFUSION_ITERATIONS: ParameterRange<u32> = ParameterRange::new(1, 50, 1);

/// Gradient anisotropic diffusion time step Δt.
///
/// The upper bound is the Δt ≤ 1/6 that
/// [`FilterKind::GradientAnisotropicDiffusion`](super::FilterKind::GradientAnisotropicDiffusion)
/// requires; the egui panel rounded it up to 0.1667, which exceeds it.
pub const ANISOTROPIC_DIFFUSION_TIME_STEP: ParameterRange<f32> =
    ParameterRange::new(0.01, 1.0 / 6.0, 0.005);

/// Gradient anisotropic diffusion conductance K.
pub const ANISOTROPIC_DIFFUSION_CONDUCTANCE: ParameterRange<f32> =
    ParameterRange::new(0.1, 100.0, 0.1);

/// Multi-Otsu class count K.
pub const MULTI_OTSU_CLASSES: ParameterRange<u32> = ParameterRange::new(2, 8, 1);

/// Curvature flow iteration count.
pub const CURVATURE_FLOW_ITERATIONS: ParameterRange<u32> = ParameterRange::new(1, 50, 1);

/// Curvature flow time step Δt.
pub const CURVATURE_FLOW_TIME_STEP: ParameterRange<f32> = ParameterRange::new(0.001, 0.166, 0.001);

/// Confidence-connected interval multiplier k of k·σ.
pub const CONFIDENCE_CONNECTED_MULTIPLIER: ParameterRange<f32> =
    ParameterRange::new(0.5, 10.0, 0.1);

/// Confidence-connected region-growing iteration limit.
pub const CONFIDENCE_CONNECTED_MAX_ITERATIONS: ParameterRange<u32> = ParameterRange::new(1, 100, 1);

/// Neighborhood-connected half-radius along each axis, in voxels.
pub const NEIGHBORHOOD_CONNECTED_RADIUS: ParameterRange<usize> = ParameterRange::new(0, 5, 1);
