//! Tests for [`ForwardFftFilter`].
//!
//! # Verified mathematical properties
//!
//! 1. Shape contract: `[H, W]` input → `[H, 2*W]` output; `[D, H, W]` → `[D, H, 2*W]`.
//! 2. DC component: for a constant image f(x,y) = v,
//!    Re(F[0,0]) = H·W·v  and  Im(F[0,0]) = 0.
//!    Proof: F(0,0) = Σ_{x,y} f(x,y)·e^{-2πi·0} = Σ f(x,y) = H·W·v.
//! 3. Parseval (unnormalized DFT): Σ_{u,v} |F[u,v]|² = H·W · Σ_{x,y} |f[x,y]|².
//!    Reference: DFT Parseval's theorem.
//! 4. Zero input: f(x,y) = 0 ⟹ F(u,v) = 0 for all u, v.
//!    Proof: F(u,v) = Σ 0·e^{...} = 0.

use crate::fft::ForwardFftFilter;
use ritk_core::filter::ops::extract_vec;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// Build a 2-D real image with row-major data and shape `[h, w]`.
fn make_real_2d(data: Vec<f32>, h: usize, w: usize) -> Image<B, 2> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new([h, w]));
    let tensor = burn::tensor::Tensor::<B, 2>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
}

/// Build a 3-D real image with row-major data and shape `[d, h, w]`.
fn make_real_3d(data: Vec<f32>, d: usize, h: usize, w: usize) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new([d, h, w]));
    let tensor = burn::tensor::Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
}

// ── Shape contract ─────────────────────────────────────────────────────────────

/// Shape contract 2-D: input `[H, W]` → output `[H, 2*W]`.
#[test]
fn output_shape_2d() {
    let img = make_real_2d(vec![0.0_f32; 4 * 6], 4, 6);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        freq.shape(),
        [4, 12],
        "2-D forward FFT output shape must be [H, 2*W]"
    );
}

/// Shape contract 3-D: input `[D, H, W]` → output `[D, H, 2*W]`.
#[test]
fn output_shape_3d() {
    let img = make_real_3d(vec![0.0_f32; 3 * 4 * 6], 3, 4, 6);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        freq.shape(),
        [3, 4, 12],
        "3-D forward FFT output shape must be [D, H, 2*W]"
    );
}

// ── DC component ───────────────────────────────────────────────────────────────

/// For a constant image f(x,y) = v, F(0,0) = H·W·v and Im(F[0,0]) = 0.
///
/// Proof: F(0,0) = Σ_{x,y} f(x,y)·e^{-2πi·0} = Σ f(x,y) = H·W·v.
/// The imaginary part vanishes because all phases are zero at u=0, v=0.
#[test]
fn dc_component_2d() {
    let h = 4_usize;
    let w = 4_usize;
    let v = 3.0_f32;
    let img = make_real_2d(vec![v; h * w], h, w);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    let (vals, _) = extract_vec(&freq).unwrap();

    // Output layout: data[r * 2*W + 2*c] = Re(F[r,c]), data[r * 2*W + 2*c + 1] = Im(F[r,c]).
    // F[0,0] occupies indices 0 (Re) and 1 (Im).
    let dc_re = vals[0];
    let dc_im = vals[1];
    let expected = (h * w) as f32 * v; // 16 * 3.0 = 48.0

    assert!(
        (dc_re - expected).abs() < 1e-3,
        "Re(F[0,0]) must equal H*W*v = {expected:.4}, got {dc_re:.6}"
    );
    assert!(
        dc_im.abs() < 1e-3,
        "Im(F[0,0]) must be 0 for constant real input, got {dc_im:.6}"
    );
}

// ── Parseval's theorem ─────────────────────────────────────────────────────────

/// Parseval's theorem (unnormalized DFT): Σ|F[u,v]|² = H·W · Σ|f[x,y]|².
///
/// Reference: DFT Parseval's theorem for the unnormalized forward transform.
#[test]
fn energy_scaling_2d() {
    let h = 4_usize;
    let w = 5_usize;
    let data: Vec<f32> = (0..h * w).map(|i| (i as f32 * 0.37_f32).sin()).collect();
    let img = make_real_2d(data.clone(), h, w);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    let (vals, _) = extract_vec(&freq).unwrap();

    // Spatial energy: Σ|f[x,y]|².
    let spatial_energy: f32 = data.iter().map(|&x| x * x).sum();

    // Spectral energy: Σ(Re² + Im²) over all W frequency bins across all H rows.
    let spectral_energy: f32 = vals
        .chunks_exact(2)
        .map(|pair| pair[0] * pair[0] + pair[1] * pair[1])
        .sum();

    let expected = (h * w) as f32 * spatial_energy;
    // Guard against near-zero denominator on degenerate inputs.
    let denom = expected.abs().max(1e-9_f32);
    let relative_err = (spectral_energy - expected).abs() / denom;

    assert!(
        relative_err < 1e-4,
        "Parseval: Σ|F|² must equal H*W·Σ|f|². \
         expected={expected:.6}, got={spectral_energy:.6}, rel_err={relative_err:.2e}"
    );
}

// ── Zero input ─────────────────────────────────────────────────────────────────

/// F(u,v) = 0 for all u,v when f(x,y) = 0.
///
/// Proof: F(u,v) = Σ_{x,y} 0·e^{...} = 0.
#[test]
fn zero_image_2d() {
    let h = 3_usize;
    let w = 4_usize;
    let img = make_real_2d(vec![0.0_f32; h * w], h, w);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    let (vals, _) = extract_vec(&freq).unwrap();

    assert_eq!(
        vals.len(),
        h * 2 * w,
        "Output buffer length must be H * 2 * W = {}",
        h * 2 * w
    );
    for (i, &x) in vals.iter().enumerate() {
        assert!(
            x.abs() < 1e-6_f32,
            "Zero-input FFT output must be zero at index {i}: got {x:.9}"
        );
    }
}
