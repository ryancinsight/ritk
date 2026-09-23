//! Series driver: validation, slice gathering along the acquisition plane,
//! and per-slice correction.

use super::slice::SliceUnringer;
use super::{GibbsError, SliceAxis, TvWindow};
use apollo_fft::application::execution::kernel::FftPrecision;
use eunomia::{Complex, NumericElement, RealField};
use std::num::NonZeroU16;

/// Shifts per side in the `mrdegibbs` reference implementation: 20, so the
/// candidate shifts step by `1/40` voxel across `[−½, ½]`.
const DEFAULT_HALF_SHIFTS: NonZeroU16 = match NonZeroU16::new(20) {
    Some(count) => count,
    None => unreachable!(),
};

/// Longest line the phase-ramp index arithmetic represents exactly.
const MAX_LINE: usize = i32::MAX as usize;

/// Gibbs-ringing removal by local subvoxel shifts over a series of volumes.
///
/// See the [module documentation](super) for the method.
///
/// # Examples
///
/// ```
/// use ritk_filter::gibbs_unringing::GibbsUnringer;
///
/// // A band-limited slice (one cosine period along the columns) is left
/// // unchanged up to the linear back-interpolation error A·ω²/8.
/// let (rows, cols) = (12, 16);
/// let omega = 2.0 * std::f64::consts::PI / cols as f64;
/// let slice: Vec<f64> = (0..rows * cols)
///     .map(|i| (omega * (i % cols) as f64).cos())
///     .collect();
///
/// let out = GibbsUnringer::default().unring([1, rows, cols], &[slice.as_slice()])?;
/// let bound = omega * omega / 8.0 + 1e-12;
/// for (a, b) in slice.iter().zip(&out[0]) {
///     assert!((a - b).abs() <= bound);
/// }
/// # Ok::<(), ritk_filter::gibbs_unringing::GibbsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GibbsUnringer {
    half_shifts: NonZeroU16,
    window: TvWindow,
    slice_axis: SliceAxis,
}

impl Default for GibbsUnringer {
    /// 20 shifts per side, `K = [1, 3]`, slices stacked along axis 0.
    fn default() -> Self {
        Self {
            half_shifts: DEFAULT_HALF_SHIFTS,
            window: TvWindow::default(),
            slice_axis: SliceAxis::default(),
        }
    }
}

impl GibbsUnringer {
    /// Use `half_shifts = M` candidate shifts per side, spaced `1/(2M)` voxel
    /// (Eq. 2).
    #[must_use]
    pub fn with_half_shifts(self, half_shifts: NonZeroU16) -> Self {
        Self {
            half_shifts,
            ..self
        }
    }

    /// Use the total-variation window `window`.
    #[must_use]
    pub fn with_window(self, window: TvWindow) -> Self {
        Self { window, ..self }
    }

    /// Correct slices stacked along `slice_axis`.
    #[must_use]
    pub fn with_slice_axis(self, slice_axis: SliceAxis) -> Self {
        Self { slice_axis, ..self }
    }

    /// Shifts per side.
    #[must_use]
    pub fn half_shifts(&self) -> NonZeroU16 {
        self.half_shifts
    }

    /// The total-variation window.
    #[must_use]
    pub fn window(&self) -> TvWindow {
        self.window
    }

    /// The through-plane axis.
    #[must_use]
    pub fn slice_axis(&self) -> SliceAxis {
        self.slice_axis
    }

    /// Remove Gibbs ringing from each volume of a series sharing `shape`
    /// (row-major, last axis fastest). Volumes and slices are corrected
    /// independently; a shape holding no samples returns the (empty) volumes
    /// as given.
    ///
    /// # Errors
    ///
    /// - [`GibbsError::VolumeLength`] when a volume's length disagrees with `shape`.
    /// - [`GibbsError::LineTooShort`] when an in-plane axis is shorter than
    ///   [`TvWindow::minimum_line`].
    /// - [`GibbsError::LineTooLong`] when an in-plane axis exceeds `i32::MAX`.
    /// - [`GibbsError::NonFinite`] for a NaN or infinite sample.
    pub fn unring<T>(&self, shape: [usize; 3], volumes: &[&[T]]) -> Result<Vec<Vec<T>>, GibbsError>
    where
        T: RealField,
        Complex<T>: FftPrecision,
    {
        let voxel_count: usize = shape.iter().product();
        for (volume, samples) in volumes.iter().enumerate() {
            if samples.len() != voxel_count {
                return Err(GibbsError::VolumeLength {
                    volume,
                    len: samples.len(),
                    expected: voxel_count,
                    shape,
                });
            }
        }
        if voxel_count == 0 {
            return Ok(volumes.iter().map(|samples| samples.to_vec()).collect());
        }
        let [row_axis, col_axis] = self.slice_axis.plane();
        let minimum = self.window.minimum_line();
        for axis in [row_axis, col_axis] {
            let len = shape[axis];
            if len < minimum {
                return Err(GibbsError::LineTooShort { axis, len, minimum });
            }
            if len > MAX_LINE {
                return Err(GibbsError::LineTooLong {
                    axis,
                    len,
                    maximum: MAX_LINE,
                });
            }
        }
        for (volume, samples) in volumes.iter().enumerate() {
            if let Some(sample) = samples.iter().position(|value| !value.is_finite()) {
                return Err(GibbsError::NonFinite { volume, sample });
            }
        }

        let strides = [shape[1] * shape[2], shape[2], 1];
        let slice_axis = self.slice_axis.index();
        let (rows, cols) = (shape[row_axis], shape[col_axis]);
        let mut workspace = SliceUnringer::new(rows, cols, self.half_shifts.get());
        let mut slice = vec![<T as NumericElement>::ZERO; rows * cols];
        let mut output = Vec::with_capacity(volumes.len());
        for samples in volumes {
            let mut corrected = vec![<T as NumericElement>::ZERO; voxel_count];
            for index in 0..shape[slice_axis] {
                let base = index * strides[slice_axis];
                let offset = |i: usize| {
                    base + (i / cols) * strides[row_axis] + (i % cols) * strides[col_axis]
                };
                for (i, value) in slice.iter_mut().enumerate() {
                    *value = samples[offset(i)];
                }
                workspace.unring(&mut slice, self.window);
                for (i, &value) in slice.iter().enumerate() {
                    corrected[offset(i)] = value;
                }
            }
            output.push(corrected);
        }
        Ok(output)
    }
}
