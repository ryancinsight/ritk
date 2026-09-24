//! Series driver: validation, slice gathering along the acquisition plane,
//! and parallel per-slice correction.

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
    /// independently, in parallel, with a result bitwise identical to a
    /// sequential pass; a shape holding no samples returns the (empty) volumes
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

        Ok(self.correct_slices::<moirai::Parallel, T>(shape, volumes))
    }

    /// Correct every slice of a validated series under execution policy `P`.
    ///
    /// Slices are gathered into a slice-major staging buffer, corrected in
    /// place in parallel — each worker reusing one [`SliceUnringer`] — and
    /// scattered back. A slice's correction reads only that slice, and the
    /// workspace carries no state between slices, so the result is bitwise
    /// independent of `P` and of which worker corrects which slice.
    pub(super) fn correct_slices<P, T>(&self, shape: [usize; 3], volumes: &[&[T]]) -> Vec<Vec<T>>
    where
        P: moirai::ExecutionPolicy,
        T: RealField,
        Complex<T>: FftPrecision,
    {
        let voxel_count: usize = shape.iter().product();
        let [row_axis, col_axis] = self.slice_axis.plane();
        let strides = [shape[1] * shape[2], shape[2], 1];
        let slice_axis = self.slice_axis.index();
        let (rows, cols) = (shape[row_axis], shape[col_axis]);
        let slice_len = rows * cols;
        let offset = |slice: usize, i: usize| {
            slice * strides[slice_axis]
                + (i / cols) * strides[row_axis]
                + (i % cols) * strides[col_axis]
        };

        // Slice `s` of volume `v` is chunk `v · slices + s` of `staged`.
        let mut staged = vec![<T as NumericElement>::ZERO; volumes.len() * voxel_count];
        for (samples, volume) in volumes.iter().zip(staged.chunks_exact_mut(voxel_count)) {
            for (index, slice) in volume.chunks_exact_mut(slice_len).enumerate() {
                for (i, value) in slice.iter_mut().enumerate() {
                    *value = samples[offset(index, i)];
                }
            }
        }
        let (half_shifts, window) = (self.half_shifts.get(), self.window);
        moirai::for_each_chunk_mut_with_state::<P, _, _, _, _>(
            &mut staged,
            slice_len,
            || SliceUnringer::new(rows, cols, half_shifts),
            |workspace, slice| workspace.unring(slice, window),
        );
        staged
            .chunks_exact(voxel_count)
            .map(|volume| {
                let mut corrected = vec![<T as NumericElement>::ZERO; voxel_count];
                for (index, slice) in volume.chunks_exact(slice_len).enumerate() {
                    for (i, &value) in slice.iter().enumerate() {
                        corrected[offset(index, i)] = value;
                    }
                }
                corrected
            })
            .collect()
    }
}
