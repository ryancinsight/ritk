use super::StochasticFractalDimensionFilter;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// A spatially-varying (non-constant) image produces an all-finite fractal-
/// dimension field — the regression denominator never degenerates and no bin
/// mean is zero in the interior.
#[test]
fn fractal_dimension_is_finite_on_textured_image() {
    let (nz, ny, nx) = (8usize, 8, 8);
    let n = nz * ny * nx;
    let mut vals = vec![0.0f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Non-separable texture so neighbourhood intensity differences
                // are non-zero in every direction.
                vals[iz * ny * nx + iy * nx + ix] =
                    ((ix as f32 * 0.7).sin() + (iy as f32 * 1.1).cos() + (iz as f32 * 0.5).sin())
                        * 30.0
                        + 50.0;
            }
        }
    }
    let img = make(vals, [nz, ny, nx]);
    let out = StochasticFractalDimensionFilter::default().apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|v| v.is_finite()),
        "fractal dimension must be finite on a textured image"
    );
    // The result is a genuine field, not a constant: the texture varies the
    // local scaling exponent.
    let (min, max) = ov
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    assert!(
        max - min > 1e-4,
        "expected a varying field, got near-constant"
    );
}

/// The estimated dimension `D = 3 − slope` is invariant under an affine
/// intensity rescale `I → a·I + b` (a > 0): scaling the mean absolute
/// difference by `a` adds the constant `ln a` to every regression `y`, shifting
/// the intercept but leaving the slope — and thus `D` — unchanged. This pins the
/// least-squares math value-semantically without a hand-rolled reference.
#[test]
fn fractal_dimension_is_affine_intensity_invariant() {
    let (nz, ny, nx) = (7usize, 9, 6);
    let n = nz * ny * nx;
    let mut vals = vec![0.0f32; n];
    for (i, v) in vals.iter_mut().enumerate() {
        *v = ((i as f32 * 0.37).sin() * (i as f32 * 0.11).cos()) * 20.0 + 40.0;
    }
    let img = make(vals.clone(), [nz, ny, nx]);
    let (a, b) = (2.5f32, 7.0f32);
    let scaled: Vec<f32> = vals.iter().map(|&v| a * v + b).collect();
    let img_scaled = make(scaled, [nz, ny, nx]);

    let f = StochasticFractalDimensionFilter::default();
    let (base, _) = extract_vec_infallible(&f.apply(&img));
    let (rescaled, _) = extract_vec_infallible(&f.apply(&img_scaled));

    for (i, (&p, &q)) in base.iter().zip(rescaled.iter()).enumerate() {
        if p.is_finite() && q.is_finite() {
            assert!(
                (p - q).abs() < 1e-3,
                "voxel {i}: D shifted under affine rescale: {p} vs {q}"
            );
        }
    }
}

/// Output geometry equals the input geometry (per-voxel transform).
#[test]
fn fractal_dimension_preserves_geometry() {
    let dims = [5usize, 5, 5];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).sin()).collect();
    let img = make(vals, dims);
    let out = StochasticFractalDimensionFilter::new([1, 1, 1]).apply(&img);
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
