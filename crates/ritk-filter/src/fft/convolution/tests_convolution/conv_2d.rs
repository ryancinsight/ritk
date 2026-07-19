//! Tests for 2-D `FftConvolutionFilter`.

use crate::fft::FftConvolutionFilter;

use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec;

type B = coeus_core::SequentialBackend;

pub(super) fn make_image_2d(vals: Vec<f32>, h: usize, w: usize) -> Image<f32, B, 2> {
    let tensor = Tensor::<f32, B>::from_slice([h, w], &vals);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

/// Output spatial shape must equal input spatial shape ("same" convention).
///
/// Invariant:
/// âˆ€ f âˆˆ â„^{hÃ—w}, âˆ€ g âˆˆ â„^{krÃ—kc}:
/// shape(FftConvolutionFilter::new(g).apply(f)) = [h, w]
#[test]
fn output_shape_matches_input() {
    let img = make_image_2d(vec![1.0_f32; 64], 8, 8);
    let kernel = make_image_2d(vec![0.0_f32; 9], 3, 3);

    let result = FftConvolutionFilter::<B>::new(&kernel)
        .unwrap()
        .apply(&img)
        .unwrap();

    assert_eq!(
        result.shape(),
        [8_usize, 8_usize],
        "output shape must equal input shape"
    );
}

/// Convolution with the 2-D Dirac delta reproduces the input exactly.
///
/// Proof: conv(f, Î´_{(1,1)}) = f by the sifting property.
/// With kernel `g[1,1] = 1` (all other entries 0) and "same" crop at (1,1):
/// out[r,c] = Î£_{dr,dc} f[r+drâˆ’1, c+dcâˆ’1] Â· g[dr,dc] = f[r,c]
///
/// Tolerance: 1e-3. Actual f32 FFT error on a 4Ã—4 image is O(1e-6).
#[test]
fn identity_kernel_convolution() {
    #[rustfmt::skip]
    let img_vals: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let img = make_image_2d(img_vals.clone(), 4, 4);

    // 3Ã—3 Dirac delta centred at (1, 1).
    #[rustfmt::skip]
    let delta: Vec<f32> = vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let kernel = make_image_2d(delta, 3, 3);

    let result = FftConvolutionFilter::<B>::new(&kernel)
        .unwrap()
        .apply(&img)
        .unwrap();

    let (out_vals, _) = extract_vec(&result).unwrap();
    assert_eq!(
        out_vals.len(),
        img_vals.len(),
        "output element count must equal input element count"
    );
    for (i, (&got, &expected)) in out_vals.iter().zip(img_vals.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-3,
            "identity kernel at index {i}: expected {expected}, got {got:.6}"
        );
    }
}

/// Interior pixel of a constant image convolved with an all-ones kernel equals
/// the kernel sum.
///
/// Proof: for constant f = 1 and interior position (r, c) where all krÃ—kc
/// neighbours lie within the image,
/// out[r,c] = Î£_{dr=0}^{krâˆ’1} Î£_{dc=0}^{kcâˆ’1} f[r+drâˆ’âŒŠkr/2âŒ‹, c+dcâˆ’âŒŠkc/2âŒ‹]
///          = kr Â· kc = 3 Â· 3 = 9.
///
/// Uses a 16Ã—16 image (interior pixel at [8, 8]) to avoid boundary effects.
/// Tolerance: 1e-3.
#[test]
fn constant_kernel_sum() {
    let img = make_image_2d(vec![1.0_f32; 256], 16, 16);
    let kernel = make_image_2d(vec![1.0_f32; 9], 3, 3);

    let result = FftConvolutionFilter::<B>::new(&kernel)
        .unwrap()
        .apply(&img)
        .unwrap();

    let (out_vals, _) = extract_vec(&result).unwrap();

    // Interior pixel index: row 8, col 8 â†’ flat index = 8 * 16 + 8 = 136.
    let interior = out_vals[8 * 16 + 8];
    assert!(
        (interior - 9.0_f32).abs() < 1e-3,
        "interior pixel should be 9.0, got {interior:.6}"
    );
}

/// Convolving with an all-zeros kernel must produce an all-zeros output.
///
/// Proof: FFT(0) = 0; FFT(f) Â· 0 = 0; IFFT(0) = 0.
/// Expected: out[r,c] = 0 for all (r, c).
/// Tolerance: 1e-6.
#[test]
fn zero_kernel_gives_zero_output() {
    let img_vals: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let img = make_image_2d(img_vals, 5, 5);
    let kernel = make_image_2d(vec![0.0_f32; 9], 3, 3);

    let result = FftConvolutionFilter::<B>::new(&kernel)
        .unwrap()
        .apply(&img)
        .unwrap();

    let (out_vals, _) = extract_vec(&result).unwrap();
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "zero kernel output at index {i} should be 0.0, got {v:.8}"
        );
    }
}
