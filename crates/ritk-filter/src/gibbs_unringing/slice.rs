//! Two-dimensional unringing of one slice: the Fourier-space split between
//! the two in-plane axes (Kellner et al. 2016, Eq. 5–6) followed by the
//! one-dimensional correction of each part along its own axis.

use super::line::LineUnringer;
use super::TvWindow;
use apollo_fft::application::execution::kernel::{fft_forward, fft_inverse, FftPrecision};
use eunomia::{CastFrom, Complex, NumericElement, RealField};

/// Reusable buffers for slices of one fixed `rows × cols` shape, row-major
/// (`cols` contiguous). Rows run along the first in-plane axis.
pub(super) struct SliceUnringer<T> {
    rows: usize,
    cols: usize,
    /// Share of each Fourier coefficient given to the part corrected along
    /// the rows axis (`G_x` of Eq. 6 with `x` the rows axis); the part
    /// corrected along the columns axis receives `1 − G_x`.
    row_axis_weight: Vec<T>,
    along_rows: Vec<Complex<T>>,
    along_cols: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    row_line: Vec<T>,
    col_line: Vec<T>,
    row_unringer: LineUnringer<T>,
    col_unringer: LineUnringer<T>,
}

impl<T> SliceUnringer<T>
where
    T: RealField,
    Complex<T>: FftPrecision,
{
    /// Buffers for `rows × cols` slices with `half_shifts` shifts per side.
    pub(super) fn new(rows: usize, cols: usize, half_shifts: u16) -> Self {
        let row_c = axis_cosine::<T>(rows);
        let col_c = axis_cosine::<T>(cols);
        let half = <T as CastFrom<i32>>::cast_from(1) / <T as CastFrom<i32>>::cast_from(2);
        let mut row_axis_weight = Vec::with_capacity(rows * cols);
        for &cr in &row_c {
            for &cc in &col_c {
                let total = cr + cc;
                // Eq. 6 is 0/0 only at the joint Nyquist bin of an even ×
                // even slice; an even split keeps G_x + G_y = 1 there too, so
                // an artifact-free slice passes through unchanged.
                row_axis_weight.push(if total > T::ZERO { cc / total } else { half });
            }
        }
        let zero = Complex::new(T::ZERO, T::ZERO);
        Self {
            rows,
            cols,
            row_axis_weight,
            along_rows: vec![zero; rows * cols],
            along_cols: vec![zero; rows * cols],
            scratch: vec![zero; rows],
            row_line: vec![T::ZERO; rows],
            col_line: vec![T::ZERO; cols],
            row_unringer: LineUnringer::new(rows, half_shifts),
            col_unringer: LineUnringer::new(cols, half_shifts),
        }
    }

    /// Replace the row-major `slice` by its unrung version.
    pub(super) fn unring(&mut self, slice: &mut [T], window: TvWindow) {
        let (rows, cols) = (self.rows, self.cols);
        debug_assert_eq!(
            slice.len(),
            rows * cols,
            "slice length must match the workspace"
        );
        for (bin, &value) in self.along_rows.iter_mut().zip(slice.iter()) {
            *bin = Complex::new(value, T::ZERO);
        }
        transform_2d(
            &mut self.along_rows,
            cols,
            &mut self.scratch,
            Direction::Forward,
        );
        for ((row_part, col_part), &weight) in self
            .along_rows
            .iter_mut()
            .zip(self.along_cols.iter_mut())
            .zip(&self.row_axis_weight)
        {
            *col_part = *row_part * (T::ONE - weight);
            *row_part *= weight;
        }
        transform_2d(
            &mut self.along_rows,
            cols,
            &mut self.scratch,
            Direction::Inverse,
        );
        transform_2d(
            &mut self.along_cols,
            cols,
            &mut self.scratch,
            Direction::Inverse,
        );

        // The weights are even in both frequencies, so each part of a real
        // slice is real up to rounding; the imaginary residue is discarded.
        for c in 0..cols {
            for (r, value) in self.row_line.iter_mut().enumerate() {
                *value = self.along_rows[r * cols + c].re;
            }
            self.row_unringer.unring(&mut self.row_line, window);
            for (r, &value) in self.row_line.iter().enumerate() {
                slice[r * cols + c] = value;
            }
        }
        for (r, out_row) in slice.chunks_exact_mut(cols).enumerate() {
            let part = &self.along_cols[r * cols..(r + 1) * cols];
            for (value, bin) in self.col_line.iter_mut().zip(part) {
                *value = bin.re;
            }
            self.col_unringer.unring(&mut self.col_line, window);
            for (out, &value) in out_row.iter_mut().zip(&self.col_line) {
                *out += value;
            }
        }
    }
}

/// `(1 + cos(2πk/n)) / 2` per frequency bin `k` of an `n`-sample axis — the
/// `1 + cos k` factor of Eq. 6 with the reference implementation's scaling,
/// which cancels in the ratio.
fn axis_cosine<T: RealField>(n: usize) -> Vec<T> {
    let n_t = <T as CastFrom<i32>>::cast_from(
        i32::try_from(n).expect("invariant: the driver bounds line lengths by i32::MAX"),
    );
    let two = <T as CastFrom<i32>>::cast_from(2);
    (0..n)
        .map(|k| {
            let k_t = <T as CastFrom<i32>>::cast_from(
                i32::try_from(k).expect("invariant: k < n <= i32::MAX"),
            );
            (T::ONE + (two * T::PI * k_t / n_t).cos()) / two
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

/// Separable 2-D transform of a row-major grid with `cols` contiguous
/// columns; the inverse is normalised by `1/(rows·cols)`.
fn transform_2d<T>(
    grid: &mut [Complex<T>],
    cols: usize,
    column: &mut [Complex<T>],
    direction: Direction,
) where
    T: NumericElement,
    Complex<T>: FftPrecision,
{
    let apply = |line: &mut [Complex<T>]| match direction {
        Direction::Forward => fft_forward(line),
        Direction::Inverse => fft_inverse(line),
    };
    for row in grid.chunks_exact_mut(cols) {
        apply(row);
    }
    for c in 0..cols {
        for (r, slot) in column.iter_mut().enumerate() {
            *slot = grid[r * cols + c];
        }
        apply(column);
        for (r, &value) in column.iter().enumerate() {
            grid[r * cols + c] = value;
        }
    }
}
