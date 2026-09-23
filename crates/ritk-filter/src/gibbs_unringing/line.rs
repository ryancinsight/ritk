//! One-dimensional unringing of a real line by local subvoxel shifts
//! (Kellner et al. 2016, Methods, "One-dimensional Case", Eq. 2–4).

use super::TvWindow;
use apollo_fft::application::execution::kernel::{fft_forward, fft_inverse, FftPrecision};
use eunomia::{CastFrom, Complex, RealField};

/// Reusable buffers for unringing lines of one fixed length.
///
/// Shift `j` of the `2M + 1` candidates moves the line by
/// `shift_j / (2M)` voxels, `shift_j ∈ {0, 1, …, M, −1, …, −M}` — the order
/// of the `mrdegibbs` reference implementation, which decides ties in favour
/// of the earlier candidate and therefore of no shift.
pub(super) struct LineUnringer<T> {
    len: usize,
    shifts: Vec<i32>,
    /// `2M` per the paper's Eq. 2 denominator.
    shift_denominator: T,
    /// `e^{iπ·shift_j·k/(n·M)}` for bins `k = 1..=last_ramp_bin`, per shift.
    ramps: Vec<Complex<T>>,
    last_ramp_bin: usize,
    spectrum: Vec<Complex<T>>,
    work: Vec<Complex<T>>,
    /// Shifted lines, `shifts.len() × len`, shift-major.
    shifted: Vec<T>,
    /// `|I_j(i + 1) − I_j(i)|` with periodic wrap, laid out like `shifted`.
    steps: Vec<T>,
}

impl<T> LineUnringer<T>
where
    T: RealField,
    Complex<T>: FftPrecision,
{
    /// Buffers for lines of `len` samples and `half_shifts = M` shifts per
    /// side. `len` is at least the window's minimum line and at most
    /// `i32::MAX`, which the driver validates.
    pub(super) fn new(len: usize, half_shifts: u16) -> Self {
        let m = i32::from(half_shifts);
        let shifts: Vec<i32> = std::iter::once(0)
            .chain(1..=m)
            .chain((1..=m).map(|s| -s))
            .collect();
        // Bins 1..=last pair with their conjugate partners n − k; an even
        // length leaves the unpaired Nyquist bin n/2.
        let last_ramp_bin = if !len.is_multiple_of(2) {
            (len - 1) / 2
        } else {
            len / 2 - 1
        };
        let n = <T as CastFrom<i32>>::cast_from(
            i32::try_from(len).expect("invariant: the driver bounds line lengths by i32::MAX"),
        );
        let m_t = <T as CastFrom<i32>>::cast_from(m);
        let mut ramps = Vec::with_capacity(shifts.len() * last_ramp_bin);
        for &shift in &shifts {
            let shift_t = <T as CastFrom<i32>>::cast_from(shift);
            for k in 1..=last_ramp_bin {
                let k_t = <T as CastFrom<i32>>::cast_from(
                    i32::try_from(k).expect("invariant: k < len <= i32::MAX"),
                );
                // Eq. 2: the phase ramp e^{i2πk·s/(2M)/n} moves the line to
                // I(x + s/(2M)) under the inverse kernel e^{+i2πkx/n}.
                let phase = T::PI * shift_t * k_t / (n * m_t);
                ramps.push(Complex::new(phase.cos(), phase.sin()));
            }
        }
        let count = shifts.len();
        Self {
            len,
            shifts,
            shift_denominator: m_t + m_t,
            ramps,
            last_ramp_bin,
            spectrum: vec![Complex::new(T::ZERO, T::ZERO); len],
            work: vec![Complex::new(T::ZERO, T::ZERO); len],
            shifted: vec![T::ZERO; count * len],
            steps: vec![T::ZERO; count * len],
        }
    }

    /// Replace `line` by its unrung version.
    ///
    /// Per voxel `x` the shift `r(x)` minimising
    /// `min(TV⁺_s(x), TV⁻_s(x))` is selected (Eq. 3–4) and the shifted line
    /// is linearly interpolated back to the grid, `I_r(x − r/(2M))`.
    pub(super) fn unring(&mut self, line: &mut [T], window: TvWindow) {
        let n = self.len;
        debug_assert_eq!(line.len(), n, "line length must match the workspace");
        for (slot, &value) in self.spectrum.iter_mut().zip(line.iter()) {
            *slot = Complex::new(value, T::ZERO);
        }
        fft_forward(&mut self.spectrum);
        self.shift_all();

        let start = usize::from(window.start());
        let end = usize::from(window.end());
        for (x, out) in line.iter_mut().enumerate() {
            let mut best = T::INFINITY;
            let mut chosen = 0;
            for j in 0..self.shifts.len() {
                let steps = &self.steps[j * n..(j + 1) * n];
                // The window never reaches past half the line (validated
                // minimum length), so each index below is distinct.
                let left = (start..=end).fold(T::ZERO, |acc, t| acc + steps[(x + n - t - 1) % n]);
                let right = (start..=end).fold(T::ZERO, |acc, t| acc + steps[(x + t) % n]);
                if left < best {
                    best = left;
                    chosen = j;
                }
                if right < best {
                    best = right;
                    chosen = j;
                }
            }
            let shifted = &self.shifted[chosen * n..(chosen + 1) * n];
            let s = <T as CastFrom<i32>>::cast_from(self.shifts[chosen]) / self.shift_denominator;
            let centre = shifted[x];
            *out = if s > T::ZERO {
                centre * (T::ONE - s) + shifted[(x + n - 1) % n] * s
            } else {
                centre * (T::ONE + s) - shifted[(x + 1) % n] * s
            };
        }
    }

    /// Fill `shifted` and `steps` from `spectrum` for every candidate shift.
    fn shift_all(&mut self) {
        let n = self.len;
        let bins = self.last_ramp_bin;
        for j in 0..self.shifts.len() {
            self.work.copy_from_slice(&self.spectrum);
            if self.shifts[j] != 0 {
                let ramps = &self.ramps[j * bins..(j + 1) * bins];
                for (k, &ramp) in (1..=bins).zip(ramps) {
                    self.work[k] *= ramp;
                    self.work[n - k] *= ramp.conj();
                }
                // The Nyquist bin of an even line has no conjugate partner; a
                // real shifted line cannot carry it (reference implementation).
                if n.is_multiple_of(2) {
                    self.work[n / 2] = Complex::new(T::ZERO, T::ZERO);
                }
            }
            fft_inverse(&mut self.work);
            let shifted = &mut self.shifted[j * n..(j + 1) * n];
            for (value, bin) in shifted.iter_mut().zip(&self.work) {
                *value = bin.re;
            }
            let steps = &mut self.steps[j * n..(j + 1) * n];
            for (i, step) in steps.iter_mut().enumerate() {
                *step = (shifted[(i + 1) % n] - shifted[i]).abs();
            }
        }
    }
}
