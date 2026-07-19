//! Tests for [`ForwardFftFilter`].
//!
//! # Verified mathematical properties
//!
//! 1. Shape contract: `[H, W]` input â†’ `[H, 2*W]` output; `[D, H, W]` â†’ `[D, H, 2*W]`.
//! 2. DC component: for a constant image f(x,y) = v,
//!    Re(F[0,0]) = HÂ·WÂ·v  and  Im(F[0,0]) = 0.
//!    Proof: F(0,0) = Î£_{x,y} f(x,y)Â·e^{-2Ï€iÂ·0} = Î£ f(x,y) = HÂ·WÂ·v.
//! 3. Parseval (unnormalized DFT): Î£_{u,v} |F[u,v]|Â² = HÂ·W Â· Î£_{x,y} |f[x,y]|Â².
//!    Reference: DFT Parseval's theorem.
//! 4. Zero input: f(x,y) = 0 âŸ¹ F(u,v) = 0 for all u, v.
//!    Proof: F(u,v) = Î£ 0Â·e^{...} = 0.

use crate::fft::ForwardFftFilter;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec;

type B = coeus_core::SequentialBackend;

/// Build a 2-D real image with row-major data and shape `[h, w]`.
fn make_real_2d(data: Vec<f32>, h: usize, w: usize) -> Image<f32, B, 2> {
    let tensor = ritk_image::tensor::Tensor::<f32, B>::from_slice([h, w], &data);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

/// Build a 3-D real image with row-major data and shape `[d, h, w]`.
fn make_real_3d(data: Vec<f32>, d: usize, h: usize, w: usize) -> Image<f32, B, 3> {
    let tensor = ritk_image::tensor::Tensor::<f32, B>::from_slice([d, h, w], &data);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

// â”€â”€ Shape contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Shape contract 2-D: input `[H, W]` â†’ output `[H, 2*W]`.
#[test]
fn output_shape_matches_input_planar() {
    let img = make_real_2d(vec![0.0_f32; 4 * 6], 4, 6);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        freq.shape(),
        [4, 12],
        "2-D forward FFT output shape must be [H, 2*W]"
    );
}

/// Shape contract 3-D: input `[D, H, W]` â†’ output `[D, H, 2*W]`.
#[test]
fn output_shape_matches_input_volume() {
    let img = make_real_3d(vec![0.0_f32; 3 * 4 * 6], 3, 4, 6);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        freq.shape(),
        [3, 4, 12],
        "3-D forward FFT output shape must be [D, H, 2*W]"
    );
}

// â”€â”€ DC component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// For a constant image f(x,y) = v, F(0,0) = HÂ·WÂ·v and Im(F[0,0]) = 0.
///
/// Proof: F(0,0) = Î£_{x,y} f(x,y)Â·e^{-2Ï€iÂ·0} = Î£ f(x,y) = HÂ·WÂ·v.
/// The imaginary part vanishes because all phases are zero at u=0, v=0.
#[test]
fn dc_component_equals_sum_of_values() {
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

// â”€â”€ Parseval's theorem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parseval's theorem (unnormalized DFT): Î£|F[u,v]|Â² = HÂ·W Â· Î£|f[x,y]|Â².
///
/// Reference: DFT Parseval's theorem for the unnormalized forward transform.
#[test]
fn energy_scales_with_normalization() {
    let h = 4_usize;
    let w = 5_usize;
    let data: Vec<f32> = (0..h * w).map(|i| (i as f32 * 0.37_f32).sin()).collect();
    let img = make_real_2d(data.clone(), h, w);
    let freq = ForwardFftFilter::new().apply(&img).unwrap();
    let (vals, _) = extract_vec(&freq).unwrap();

    // Spatial energy: Î£|f[x,y]|Â².
    let spatial_energy: f32 = data.iter().map(|&x| x * x).sum();

    // Spectral energy: Î£(ReÂ² + ImÂ²) over all W frequency bins across all H rows.
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
        "Parseval: Î£|F|Â² must equal H*WÂ·Î£|f|Â². \
         expected={expected:.6}, got={spectral_energy:.6}, rel_err={relative_err:.2e}"
    );
}

// â”€â”€ Zero input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// F(u,v) = 0 for all u,v when f(x,y) = 0.
///
/// Proof: F(u,v) = Î£_{x,y} 0Â·e^{...} = 0.
#[test]
fn all_zero_input_gives_all_zero_output() {
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

// â”€â”€ Half-Hermitian forward FFT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// The half-Hermitian forward FFT keeps the first `W/2+1` complex columns of the
/// full forward FFT, bitwise. Shape: `[D, H, W]` â†’ `[D, H, 2*(W/2+1)]`, and the
/// retained interleaved values equal the full transform's leading columns.
#[test]
fn half_hermitian_matches_full_forward_leading_columns() {
    use crate::fft::RealToHalfHermitianForwardFftFilter;
    let (d, h, w) = (2usize, 4, 8);
    let data: Vec<f32> = (0..d * h * w)
        .map(|i| (i as f32 * 0.37).sin() * 10.0)
        .collect();
    let img = make_real_3d(data, d, h, w);

    let full = ForwardFftFilter::new().apply(&img).unwrap();
    let half = RealToHalfHermitianForwardFftFilter::new()
        .apply(&img)
        .unwrap();

    let half_cols = w / 2 + 1;
    assert_eq!(
        half.shape(),
        [d, h, 2 * half_cols],
        "half-Hermitian output shape must be [D, H, 2*(W/2+1)]"
    );

    let (fv, _) = extract_vec(&full).unwrap();
    let (hv, _) = extract_vec(&half).unwrap();
    let full_row = 2 * w;
    let keep = 2 * half_cols;
    let rows = d * h;
    for r in 0..rows {
        for c in 0..keep {
            let got = hv[r * keep + c];
            let want = fv[r * full_row + c];
            assert_eq!(got, want, "half col mismatch at row {r}, elem {c}");
        }
    }
}
