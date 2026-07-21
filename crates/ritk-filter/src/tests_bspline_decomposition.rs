use super::bspline_decomposition;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// Whole-sample mirror index into `[0, n)`, period `2(n-1)`.
fn mirror(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= n as isize {
        m = period - m;
    }
    m as usize
}

/// Reconstructing the coefficients through the cubic basis at the grid points
/// must recover the original samples: `(1/6)c[k-1] + (2/3)c[k] + (1/6)c[k+1] =
/// s[k]` along a 1-D line (z=1, y=1). This is the defining inversion property of
/// the decomposition.
#[test]
fn decomposition_inverts_cubic_basis_along_x() {
    let samples = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let n = samples.len();
    let img = make(samples.clone(), [1, 1, n]);
    let out = bspline_decomposition(&img).expect("infallible: validated precondition");
    let (c, _) = extract_vec_infallible(&out);
    for k in 0..n {
        let ki = k as isize;
        let recon = (1.0 / 6.0) * c[mirror(ki - 1, n)]
            + (2.0 / 3.0) * c[k]
            + (1.0 / 6.0) * c[mirror(ki + 1, n)];
        assert!(
            (recon - samples[k]).abs() < 1e-4,
            "reconstruction at x={k}: got {recon}, want {}",
            samples[k]
        );
    }
}

/// A constant image decomposes to the same constant (the cubic basis has unit
/// DC gain), exactly.
#[test]
fn constant_image_is_fixed_point() {
    let img = make(vec![7.0f32; 3 * 4 * 5], [3, 4, 5]);
    let out = bspline_decomposition(&img).expect("infallible: validated precondition");
    let (c, _) = extract_vec_infallible(&out);
    assert!(
        c.iter().all(|&v| (v - 7.0).abs() < 1e-3),
        "constant image must decompose to the same constant"
    );
}

/// Output shape matches input.
#[test]
fn output_shape_preserved() {
    let img = make(vec![0.0f32; 2 * 3 * 4], [2, 3, 4]);
    let out = bspline_decomposition(&img).expect("infallible: validated precondition");
    assert_eq!(out.shape(), [2, 3, 4]);
}
