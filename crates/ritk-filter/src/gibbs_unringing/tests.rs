use super::{GibbsError, GibbsUnringer, SliceAxis, TvWindow};
use apollo_fft::application::execution::kernel::FftPrecision;
use eunomia::{Complex, NumericElement, RealField};
use std::f64::consts::PI;

/// Samples at integer `x` of the periodic box `[lo, hi)` of period `n`,
/// band-limited to the frequencies `|k| < n/2`:
/// `w/n + Σ_{k=1}^{n/2−1} (sin(2πk(x−lo)/n) − sin(2πk(x−hi)/n)) / (πk)`.
///
/// This is the image an MRI scanner reconstructs from the truncated k-space of
/// an ideal edge pair; with integer `lo` and `hi` every sample falls on an
/// extremum of the sinc ringing (Kellner et al. 2016, Fig. 1A), the worst case.
fn truncated_box(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let nf = n as f64;
    (0..n)
        .map(|x| {
            let x = x as f64;
            (1..n / 2).fold((hi - lo) / nf, |acc, k| {
                let k = k as f64;
                acc + ((2.0 * PI * k * (x - lo) / nf).sin() - (2.0 * PI * k * (x - hi) / nf).sin())
                    / (PI * k)
            })
        })
        .collect()
}

/// Total variation `Σ_{k=2}^{8} |v[e+k+1] − v[e+k]|` on the flat side of the
/// edge at index `e`, excluding the transition voxels `e−1, e, e+1`.
fn flat_side_tv(v: &[f64], e: usize) -> f64 {
    (2..=8).map(|k| (v[e + k + 1] - v[e + k]).abs()).sum()
}

fn to_scalar<T: RealField>(values: &[f64]) -> Vec<T> {
    values.iter().map(|&v| T::from_f64(v)).collect()
}

fn to_f64<T: RealField>(values: &[T]) -> Vec<f64> {
    values.iter().map(|&v| NumericElement::to_f64(v)).collect()
}

/// A step edge sampled from truncated k-space rings next to the edge, and
/// unringing reduces that oscillation by at least `4π`.
///
/// Before: with the edge on the grid the samples at distance `k` sit on the
/// extrema of `Si(πk)/π`, whose ringing is `(−1)^{k+1}/(π²k) + O(1/k³)`
/// (the asymptotic expansion `Si(x) − π/2 = −cos x/x − sin x/x² + O(x⁻³)` at
/// `sin πk = 0`), so the flat-side variation over `k = 2..=8` is to leading
/// order `Σ (1/π²)(1/k + 1/(k+1)) ≈ 0.33`; the test demands at least half of
/// that, leaving the remainder terms and the opposite edge's tail
/// (`≲ 1/(π²·56)` per sample at `n = 128`) inside the margin.
///
/// After: the half-voxel shift samples the ringing at its zero crossings,
/// where `Si(x) − π/2 = ∓1/x² + O(x⁻³)`, and the back-interpolation averages
/// two such samples of opposite sign, leaving `O(1/(π³k³))`. The per-voxel
/// ratio of the two orders is `πk² ≥ 4π` over the window, so the variation
/// must drop by at least `4π ≈ 12.6`.
fn step_edge_ringing_drops<T>()
where
    T: RealField,
    Complex<T>: FftPrecision,
{
    let (rows, cols) = (16, 128);
    let (lo, hi) = (32_usize, 96_usize);
    let line = truncated_box(cols, lo as f64, hi as f64);
    let slice: Vec<f64> = (0..rows).flat_map(|_| line.iter().copied()).collect();
    let input = to_scalar::<T>(&slice);

    let leading_order: f64 = (2..=8)
        .map(|k| {
            let k = f64::from(k);
            (1.0 / k + 1.0 / (k + 1.0)) / (PI * PI)
        })
        .sum();
    let before = flat_side_tv(&line, lo);
    assert!(
        before >= 0.5 * leading_order,
        "the synthesized edge must ring: {before} < half of {leading_order}"
    );

    let out = GibbsUnringer::default()
        .unring([1, rows, cols], &[input.as_slice()])
        .expect("valid slice");
    let out = to_f64(&out[0]);
    let factor = 4.0 * PI;
    for r in 0..rows {
        let row = &out[r * cols..(r + 1) * cols];
        let after = flat_side_tv(row, lo);
        assert!(
            after * factor <= before,
            "row {r}: flat-side variation {after} did not drop by {factor} from {before}"
        );
        // The mirrored flat side of the falling edge at `hi`.
        let mirrored: f64 = (2..=8).map(|k| (row[hi - k - 1] - row[hi - k]).abs()).sum();
        let before_mirrored: f64 = (2..=8)
            .map(|k| (line[hi - k - 1] - line[hi - k]).abs())
            .sum();
        assert!(
            mirrored * factor <= before_mirrored,
            "row {r}: mirrored variation {mirrored} did not drop by {factor} from {before_mirrored}"
        );
    }
}

#[test]
fn step_edge_ringing_drops_f32() {
    step_edge_ringing_drops::<f32>();
}

#[test]
fn step_edge_ringing_drops_f64() {
    step_edge_ringing_drops::<f64>();
}

/// A plane wave well inside the band returns within the linear
/// back-interpolation bound `A·max(ω_r², ω_c²)/8` (module docs: each split
/// part is the same plane wave scaled by `G_x` or `G_y`, `G_x + G_y = 1`, and
/// its one-dimensional pass errs by at most its amplitude times `ω²/8` along
/// its axis) plus FFT rounding: four transform passes over at most
/// `log₂ 32 = 5` butterfly levels each, each level contributing at most a few
/// `ε` of the amplitude, budgeted as `64·ε·A`.
fn band_limited_slice_returns<T>()
where
    T: RealField,
    Complex<T>: FftPrecision,
{
    let (rows, cols) = (24, 32);
    let amplitude = 3.0;
    let (omega_r, omega_c) = (2.0 * PI / rows as f64, 2.0 * PI * 2.0 / cols as f64);
    let slice: Vec<f64> = (0..rows * cols)
        .map(|i| {
            let (r, c) = ((i / cols) as f64, (i % cols) as f64);
            amplitude * (omega_r * r + omega_c * c + 0.4).cos()
        })
        .collect();
    let input = to_scalar::<T>(&slice);
    let out = GibbsUnringer::default()
        .unring([1, rows, cols], &[input.as_slice()])
        .expect("valid slice");
    let interpolation = amplitude * omega_r.max(omega_c).powi(2) / 8.0;
    let rounding = 64.0 * NumericElement::to_f64(T::EPSILON) * amplitude;
    let bound = interpolation + rounding;
    let exact = to_f64(&input);
    for (i, (&a, b)) in exact.iter().zip(to_f64(&out[0])).enumerate() {
        assert!((a - b).abs() <= bound, "sample {i}: |{a} − {b}| > {bound}");
    }
}

#[test]
fn band_limited_slice_returns_f32() {
    band_limited_slice_returns::<f32>();
}

#[test]
fn band_limited_slice_returns_f64() {
    band_limited_slice_returns::<f64>();
}

/// A constant slice has zero variation at every shift, so the unshifted
/// candidate wins every tie and the slice returns up to FFT rounding.
#[test]
fn constant_slice_is_preserved() {
    let slice = vec![7.25_f64; 10 * 11];
    let out = GibbsUnringer::default()
        .unring([1, 10, 11], &[slice.as_slice()])
        .expect("valid slice");
    for &v in &out[0] {
        assert!((v - 7.25).abs() <= 64.0 * f64::EPSILON * 7.25, "{v}");
    }
}

/// A deterministic, non-separable test volume with sharp edges.
fn edgy_volume(shape: [usize; 3], seed: usize) -> Vec<f64> {
    (0..shape.iter().product::<usize>())
        .map(|i| {
            let (z, y, x) = (
                i / (shape[1] * shape[2]),
                (i / shape[2]) % shape[1],
                i % shape[2],
            );
            let inside = (x + seed) % 7 < 3 || (y * 3 + z + seed) % 11 < 4;
            let base = if inside { 10.0 } else { 1.0 };
            base + ((i * 2_654_435_761 + seed) % 97) as f64 / 97.0
        })
        .collect()
}

/// The output has the input's shape, and a series equals its volumes
/// processed alone, bitwise: volumes share no state.
#[test]
fn volumes_are_independent() {
    let shape = [3, 12, 14];
    let volumes: Vec<Vec<f64>> = (0..3).map(|seed| edgy_volume(shape, seed)).collect();
    let views: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let unringer = GibbsUnringer::default();
    let series = unringer.unring(shape, &views).expect("valid series");
    assert_eq!(series.len(), volumes.len());
    for (volume, together) in volumes.iter().zip(&series) {
        assert_eq!(together.len(), volume.len());
        let alone = unringer
            .unring(shape, &[volume.as_slice()])
            .expect("valid volume");
        assert_eq!(&alone[0], together);
        assert_ne!(together, volume, "the edges must be corrected");
    }
}

/// Slices are corrected independently, and the slice axis selects the plane:
/// the same data stacked along axis 2 gives the same slices bitwise.
#[test]
fn slices_follow_the_slice_axis() {
    let shape = [3, 12, 14];
    let volume = edgy_volume(shape, 5);
    let unringer = GibbsUnringer::default();
    let whole = unringer
        .unring(shape, &[volume.as_slice()])
        .expect("valid volume");
    let plane = shape[1] * shape[2];
    for z in 0..shape[0] {
        let slice = &volume[z * plane..(z + 1) * plane];
        let alone = unringer.unring([1, 12, 14], &[slice]).expect("valid slice");
        assert_eq!(alone[0].as_slice(), &whole[0][z * plane..(z + 1) * plane]);
    }

    // Transpose [z][y][x] -> [y][x][z]; its axis-2 slices are the axis-0
    // slices of the original.
    let moved_shape = [12, 14, 3];
    let mut moved = vec![0.0; volume.len()];
    for (i, &v) in volume.iter().enumerate() {
        let (z, y, x) = (i / plane, (i / 14) % 12, i % 14);
        moved[(y * 14 + x) * 3 + z] = v;
    }
    let moved_out = unringer
        .with_slice_axis(SliceAxis::Axis2)
        .unring(moved_shape, &[moved.as_slice()])
        .expect("valid volume");
    for (i, &v) in whole[0].iter().enumerate() {
        let (z, y, x) = (i / plane, (i / 14) % 12, i % 14);
        assert_eq!(moved_out[0][(y * 14 + x) * 3 + z], v);
    }
}

#[test]
fn short_lines_are_rejected() {
    let unringer = GibbsUnringer::default();
    assert_eq!(TvWindow::default().minimum_line(), 9);
    let short = vec![1.0_f64; 12 * 8];
    assert_eq!(
        unringer.unring([1, 12, 8], &[short.as_slice()]),
        Err(GibbsError::LineTooShort {
            axis: 2,
            len: 8,
            minimum: 9
        })
    );
    let short_rows = vec![1.0_f64; 8 * 12];
    assert_eq!(
        unringer.unring([1, 8, 12], &[short_rows.as_slice()]),
        Err(GibbsError::LineTooShort {
            axis: 1,
            len: 8,
            minimum: 9
        })
    );
    // The through-plane axis may be as short as one slice; nine in-plane
    // samples are enough.
    let exact = vec![2.0_f64; 9 * 9];
    let out = unringer
        .unring([1, 9, 9], &[exact.as_slice()])
        .expect("minimum lines");
    assert_eq!(out[0].len(), 81);
}

#[test]
fn empty_shapes_and_series_pass_through() {
    let unringer = GibbsUnringer::default();
    let empty: Vec<f64> = Vec::new();
    for shape in [[0, 12, 12], [2, 0, 12], [2, 12, 0]] {
        assert_eq!(
            unringer.unring(shape, &[empty.as_slice(), empty.as_slice()]),
            Ok(vec![Vec::new(), Vec::new()])
        );
    }
    assert_eq!(unringer.unring::<f64>([1, 12, 12], &[]), Ok(Vec::new()));
}

#[test]
fn non_finite_samples_are_rejected() {
    let unringer = GibbsUnringer::default();
    let clean = vec![1.0_f64; 10 * 10];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut dirty = clean.clone();
        dirty[37] = bad;
        assert_eq!(
            unringer.unring([1, 10, 10], &[clean.as_slice(), dirty.as_slice()]),
            Err(GibbsError::NonFinite {
                volume: 1,
                sample: 37
            })
        );
    }
}

#[test]
fn volume_length_mismatch_is_rejected() {
    let short = vec![1.0_f64; 99];
    assert_eq!(
        GibbsUnringer::default().unring([1, 10, 10], &[short.as_slice()]),
        Err(GibbsError::VolumeLength {
            volume: 0,
            len: 99,
            expected: 100,
            shape: [1, 10, 10],
        })
    );
}

#[test]
fn window_bounds_are_validated() {
    assert_eq!(
        TvWindow::new(4, 3),
        Err(GibbsError::InvalidWindow { start: 4, end: 3 })
    );
    let single = TvWindow::new(0, 0).expect("a one-term window is valid");
    assert_eq!(single.minimum_line(), 3);
    let wide = TvWindow::new(1, 5).expect("valid window");
    let slice = vec![1.0_f64; 12 * 12];
    assert_eq!(
        GibbsUnringer::default()
            .with_window(wide)
            .unring([1, 12, 12], &[slice.as_slice()]),
        Err(GibbsError::LineTooShort {
            axis: 1,
            len: 12,
            minimum: 13
        })
    );
}
