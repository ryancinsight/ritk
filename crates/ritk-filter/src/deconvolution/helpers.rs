//! Internal FFT-based convolution helpers for deconvolution filters.
//!
//! # Design
//!
//! The const-generic `convolve<const D: usize>` implements linear convolution
//! via FFT with "same" output cropping for any dimensionality, dispatching via
//! `fft_nd`. The padding, FFT, pointwise multiply, IFFT,
//! and crop logic is shared across all supported dimensionalities.

use crate::fft::convolution::{fft_nd, ForwardFft, InverseFft};
use rustfft::{num_complex::Complex, FftPlanner};

// ── Padding & FFT ───────────────────────────────────────────────────────────

/// Compute next-power-of-two padded dimensions for linear convolution.
///
/// Each axis is padded to `(img_dim + ker_dim - 1).next_power_of_two()`.
pub(super) fn pad_dims<const D: usize>(img_dims: &[usize; D], ker_dims: &[usize; D]) -> [usize; D] {
    let mut out = [0usize; D];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (img_dims[i] + ker_dims[i] - 1).next_power_of_two();
    }
    out
}

/// Total number of elements in the padded buffer (product of all pad dims).
pub(super) fn pad_total<const D: usize>(pad: &[usize; D]) -> usize {
    pad.iter().product()
}

/// Row-major stride for the leading `D - d` dimensions of a padded array.
///
/// For a 3-D array with shape `[pd, ph, pw]`:
/// - stride(0) = ph * pw (depth stride)
/// - stride(1) = pw (row stride)
/// - stride(2) = 1 (col stride)
fn stride<const D: usize>(pad: &[usize; D], d: usize) -> usize {
    pad[d + 1..D].iter().product()
}

/// Decode a flat row-major index into per-axis coordinates.
///
/// Returns a `[usize; D]` array where element `d` is the coordinate along
/// axis `d` of an array with shape `dims`.
pub(super) fn decode_coords<const D: usize>(flat: usize, dims: &[usize; D]) -> [usize; D] {
    let mut rem = flat;
    std::array::from_fn(|d| {
        let s = stride::<D>(dims, d);
        let coord = rem / s;
        rem %= s;
        coord
    })
}

/// Encode per-axis coordinates into a flat row-major index.
pub(super) fn encode_flat<const D: usize>(coords: &[usize; D], dims: &[usize; D]) -> usize {
    coords
        .iter()
        .enumerate()
        .map(|(d, &c)| c * stride::<D>(dims, d))
        .sum()
}

/// Place a real-valued image/kernel into a zero-padded complex buffer at
/// position (0, 0, …, 0) (corner placement, no centering).
pub(super) fn place_corner<const D: usize>(
    buf: &mut [Complex<f32>],
    vals: &[f32],
    dims: &[usize; D],
    pad: &[usize; D],
) {
    for (flat, &v) in vals.iter().enumerate() {
        let coords = decode_coords::<D>(flat, dims);
        let pflat = encode_flat::<D>(&coords, pad);
        buf[pflat] = Complex::new(v, 0.0);
    }
}

/// Place a real-valued image/kernel into a zero-padded complex buffer at
/// position `offset` (corner + offset per axis).
fn place_at_offset<const D: usize>(
    buf: &mut [Complex<f32>],
    vals: &[f32],
    dims: &[usize; D],
    pad: &[usize; D],
    offset: &[usize; D],
) {
    for (flat, &v) in vals.iter().enumerate() {
        let coords = decode_coords::<D>(flat, dims);
        let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + offset[d]);
        let pflat = encode_flat::<D>(&pcoords, pad);
        buf[pflat] = Complex::new(v, 0.0);
    }
}

/// Place a kernel into a zero-padded complex buffer with zero-phase centering.
///
/// The kernel center `(kd/2, kd/2, …)` is shifted to padded origin `(0, 0, …)`
/// via modular arithmetic (circular shift), eliminating linear-phase delay.
pub(super) fn place_centered<const D: usize>(
    buf: &mut [Complex<f32>],
    vals: &[f32],
    dims: &[usize; D],
    pad: &[usize; D],
) {
    for (flat, &v) in vals.iter().enumerate() {
        let coords = decode_coords::<D>(flat, dims);
        let pcoords: [usize; D] =
            std::array::from_fn(|d| (coords[d] + pad[d] - dims[d] / 2) % pad[d]);
        let pflat = encode_flat::<D>(&pcoords, pad);
        buf[pflat] = Complex::new(v, 0.0);
    }
}

/// Execute a forward or inverse FFT on a padded complex buffer.
pub(super) fn run_fft<const D: usize, Dir: crate::fft::convolution::FftDirection>(
    buf: &mut [Complex<f32>],
    pad: &[usize; D],
    planner: &mut FftPlanner<f32>,
) {
    fft_nd::<D, Dir>(buf, pad, planner);
}

/// Pad two real-valued arrays into complex buffers, execute forward FFT on both.
///
/// The image is placed at `img_offset` per axis; the kernel is placed with
/// zero-phase centering.
pub(super) fn pad_and_fft<const D: usize>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    ker_vals: &[f32],
    ker_dims: &[usize; D],
    pad: &[usize; D],
    pad_n: usize,
    img_offset: &[usize; D],
) -> (Vec<Complex<f32>>, Vec<Complex<f32>>) {
    let mut img_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
    place_at_offset::<D>(&mut img_padded, img_vals, img_dims, pad, img_offset);

    let mut ker_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
    place_centered::<D>(&mut ker_padded, ker_vals, ker_dims, pad);

    let mut planner = FftPlanner::<f32>::new();
    run_fft::<D, ForwardFft>(&mut img_padded, pad, &mut planner);
    run_fft::<D, ForwardFft>(&mut ker_padded, pad, &mut planner);

    (img_padded, ker_padded)
}

/// Execute inverse FFT on `buf` and crop to `out_dims`, returning real values.
///
/// `crop_offset` gives the per-axis starting position in the padded buffer from
/// which output elements are read (ITK `CropOutput` convention: `ker_dim/2` per
/// axis). The 1/N scaling factor for the inverse FFT is applied during cropping.
pub(super) fn ifft_and_crop<const D: usize>(
    buf: &mut [Complex<f32>],
    out_dims: &[usize; D],
    pad: &[usize; D],
    pad_n: usize,
    crop_offset: &[usize; D],
) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    run_fft::<D, InverseFft>(buf, pad, &mut planner);

    let scale = 1.0_f32 / pad_n as f32;
    let out_n: usize = out_dims.iter().product();
    let mut out = vec![0.0_f32; out_n];

    for (flat, o) in out.iter_mut().enumerate() {
        let coords = decode_coords::<D>(flat, out_dims);
        let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + crop_offset[d]);
        let pflat = encode_flat::<D>(&pcoords, pad);
        *o = buf[pflat].re * scale;
    }
    out
}

// ── Const-generic convolution ──────────────────────────────────────────────

/// FFT-based linear convolution returning a "same"-sized output.
///
/// # Arguments
/// - `image` — row-major image slice, shape `img_dims`
/// - `kernel` — row-major kernel slice, shape `ker_dims`
///
/// # Output
/// Row-major `Vec<f32>` of shape `img_dims` (same as input).
///
/// # Invariant
/// Output length equals the product of `img_dims`.
#[allow(dead_code)]
pub(super) fn convolve<const D: usize>(
    image: &[f32],
    img_dims: &[usize; D],
    kernel: &[f32],
    ker_dims: &[usize; D],
) -> Vec<f32> {
    let pad = pad_dims::<D>(img_dims, ker_dims);
    let pad_n = pad_total::<D>(&pad);

    let mut img_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    place_corner::<D>(&mut img_pad, image, img_dims, &pad);

    let mut ker_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    place_corner::<D>(&mut ker_pad, kernel, ker_dims, &pad);

    let mut planner = FftPlanner::<f32>::new();
    run_fft::<D, ForwardFft>(&mut img_pad, &pad, &mut planner);
    run_fft::<D, ForwardFft>(&mut ker_pad, &pad, &mut planner);

    // Pointwise multiplication in frequency domain
    for (a, b) in img_pad.iter_mut().zip(ker_pad.iter()) {
        *a = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
    }

    run_fft::<D, InverseFft>(&mut img_pad, &pad, &mut planner);

    // Crop to "same" size with center offset for convolution alignment
    let scale = 1.0_f32 / pad_n as f32;
    let out_n: usize = img_dims.iter().product();
    let mut result = vec![0.0_f32; out_n];

    for (flat, r) in result.iter_mut().enumerate() {
        let coords = decode_coords::<D>(flat, img_dims);
        let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + ker_dims[d] / 2);
        let pflat = encode_flat::<D>(&pcoords, &pad);
        *r = img_pad[pflat].re * scale;
    }
    result
}
