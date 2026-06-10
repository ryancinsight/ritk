use super::*;
use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn image_stats(vals: &[f32]) -> (f32, f32) {
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
    (mean, var.sqrt())
}

/// Uniform image → after N iterations the image is still uniform (all
/// fluxes are zero since all differences are zero).
#[test]
fn test_uniform_image_unchanged() {
    let dims = [8, 8, 8];
    let val = 42.0_f32;
    let vals = vec![val; 8 * 8 * 8];
    let img = make_image(vals, dims);
    let filter = AnisotropicDiffusionFilter::<ExponentialConductance>::new(DiffusionConfig {
        num_iterations: 20,
        ..Default::default()
    });
    let out = filter.apply(&img).unwrap();

    let (result, _) = extract_vec_infallible(&out);
    for &v in &result {
        assert!(
            (v - val).abs() < 1e-4,
            "expected {val} for uniform image after diffusion, got {v}"
        );
    }
}

/// Step-edge test: left half = 50, right half = 200.
///
/// After anisotropic diffusion with K large enough to inhibit diffusion
/// across the edge:
/// 1. Mean intensity is conserved (Neumann BC → no mass leaving domain).
/// 2. The sign of (mean_right − mean_left) remains positive.
/// 3. The mean of the left homogeneous region stays close to 50.
#[test]
fn test_step_edge_preservation() {
    let [nz, ny, nx] = [10usize, 10, 20];
    let n = nz * ny * nx;

    let mut vals = vec![0.0_f32; n];
    let initial_mean;
    {
        let mut sum = 0.0_f32;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = if ix < nx / 2 { 50.0_f32 } else { 200.0_f32 };
                    vals[iz * ny * nx + iy * nx + ix] = v;
                    sum += v;
                }
            }
        }
        initial_mean = sum / n as f32;
    }

    let img = make_image(vals.clone(), [nz, ny, nx]);
    let filter = AnisotropicDiffusionFilter::<ExponentialConductance>::new(DiffusionConfig {
        num_iterations: 10,
        conductance: 30.0, // large K → moderate edge inhibition
        ..Default::default()
    });
    let out = filter.apply(&img).unwrap();

    let (result, _) = extract_vec_infallible(&out);

    // 1. Mean conservation (Neumann BC → total mass is invariant).
    let final_mean: f32 = result.iter().sum::<f32>() / n as f32;
    assert!(
        (final_mean - initial_mean).abs() < 0.5,
        "mean should be conserved: initial={initial_mean:.4} final={final_mean:.4}"
    );

    // 2. Sign of (mean_right − mean_left) is preserved.
    let mut mean_left = 0.0_f32;
    let mut mean_right = 0.0_f32;
    let half = n / 2;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                if ix < nx / 2 {
                    mean_left += result[flat];
                } else {
                    mean_right += result[flat];
                }
            }
        }
    }
    mean_left /= half as f32;
    mean_right /= half as f32;
    assert!(
        mean_right > mean_left,
        "edge sign should be preserved: left={mean_left:.2} right={mean_right:.2}"
    );

    // 3. Left region mean stays within 5.0 of original 50.0 after 10 iters.
    assert!(
        (mean_left - 50.0).abs() < 5.0,
        "left region mean too far from 50: {mean_left:.4}"
    );
}

/// Mean conservation with any image: total intensity should be approximately
/// invariant under anisotropic diffusion with Neumann BCs.
#[test]
fn test_mean_conservation() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let n = nz * ny * nx;
    // Simple ramp image
    let vals: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 100.0).collect();
    let img = make_image(vals.clone(), [nz, ny, nx]);

    let (initial_mean, _) = image_stats(&vals);

    let filter = AnisotropicDiffusionFilter::<ExponentialConductance>::new(DiffusionConfig {
        num_iterations: 30,
        ..Default::default()
    });
    let out = filter.apply(&img).unwrap();

    let (result, _) = extract_vec_infallible(&out);
    let (final_mean, _) = image_stats(&result);

    let rel_error = (final_mean - initial_mean).abs() / (initial_mean.abs() + 1e-6);
    assert!(
        rel_error < 0.005,
        "mean not conserved: initial={initial_mean:.4} final={final_mean:.4} rel={rel_error:.6}"
    );
}

/// Quadratic conductance function also converges without blowing up.
#[test]
fn test_quadratic_conductance_stable() {
    let [nz, ny, nx] = [6usize, 6, 6];
    let n = nz * ny * nx;
    let vals: Vec<f32> = (0..n).map(|i| (i % 10) as f32 * 10.0).collect();
    let img = make_image(vals.clone(), [nz, ny, nx]);

    let filter = AnisotropicDiffusionFilter::<QuadraticConductance>::new(DiffusionConfig {
        num_iterations: 20,
        ..Default::default()
    });
    let out = filter.apply(&img).unwrap();

    let (result, _) = extract_vec_infallible(&out);

    // All values should remain finite and in a reasonable range.
    let initial_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let initial_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    for &v in &result {
        assert!(
            v.is_finite(),
            "quadratic conductance produced non-finite value"
        );
        assert!(
            v >= initial_min - 1.0 && v <= initial_max + 1.0,
            "value {v} outside initial range [{initial_min}, {initial_max}]"
        );
    }
}
