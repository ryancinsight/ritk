//! Tests for chan_vese level set segmentation.
//! Extracted to keep the 500-line structural limit.

use super::*;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Point, Spacing};
use ritk_image::test_support::{make_image, make_image_with};

type B = SequentialBackend;

/// Delegation wrapper so tests call `compute_curvature` while production code uses
/// `compute_curvature_into` directly.
fn compute_curvature(phi: &[f64], dims: [usize; 3], kappa: &mut [f64]) {
    compute_curvature_into(phi, dims, kappa);
}

fn make_image_with_metadata(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, B, 3> {
    make_image_with(
        data,
        dims,
        Some(Point::new(origin)),
        Some(Spacing::new(spacing)),
        None,
    )
}

fn get_values(image: &Image<f32, B, 3>) -> Vec<f32> {
    image.data().to_vec()
}

// ── Test 1: Bimodal sphere recovery ────────────────────────────────────────

#[test]
fn test_bimodal_sphere_segmentation() {
    // 16×16×16 image with a sphere of radius 5 at center (8,8,8).
    // Foreground intensity = 200, background intensity = 50.
    // Chan-Vese should approximately recover the sphere interior.
    let (nz, ny, nx) = (16, 16, 16);
    let n = nz * ny * nx;
    let mut data = vec![50.0_f32; n];
    let center = [8.0_f64, 8.0, 8.0];
    let radius = 5.0_f64;
    let radius_sq = radius * radius;

    let mut sphere_count = 0usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                if dz * dz + dy * dy + dx * dx <= radius_sq {
                    data[iz * ny * nx + iy * nx + ix] = 200.0;
                    sphere_count += 1;
                }
            }
        }
    }

    let image = make_image(data, [nz, ny, nx]);
    let mut cv = ChanVeseSegmentation::new();
    cv.max_iterations = 300;
    cv.dt = 0.1;
    let result = cv
        .apply(&image)
        .expect("infallible: validated precondition");
    let vals = get_values(&result);

    let seg_count: usize = vals.iter().filter(|&&v| v == 1.0).count();

    let mut intersection = 0usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iz * ny * nx + iy * nx + ix;
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                let in_sphere = dz * dz + dy * dy + dx * dx <= radius_sq;
                let in_seg = vals[idx] == 1.0;
                if in_sphere && in_seg {
                    intersection += 1;
                }
            }
        }
    }

    let dice = if sphere_count + seg_count > 0 {
        2.0 * intersection as f64 / (sphere_count + seg_count) as f64
    } else {
        0.0
    };

    assert!(
        dice > 0.5,
        "Dice coefficient {:.4} too low; Chan-Vese should recover sphere (sphere_count={}, seg_count={}, intersection={})",
        dice, sphere_count, seg_count, intersection
    );
}

// ── Test 2: Uniform image converges to homogeneous label ───────────────────

#[test]
fn test_uniform_image_homogeneous_output() {
    let dims = [8, 8, 8];
    let n: usize = dims.iter().product();
    let data = vec![100.0_f32; n];
    let image = make_image(data, dims);

    let mut cv = ChanVeseSegmentation::new();
    cv.max_iterations = 300;
    let result = cv
        .apply(&image)
        .expect("infallible: validated precondition");
    let vals = get_values(&result);

    let ones: usize = vals.iter().filter(|&&v| v == 1.0).count();
    let zeros: usize = vals.iter().filter(|&&v| v == 0.0).count();

    let majority = ones.max(zeros);
    let ratio = majority as f64 / n as f64;
    assert!(
        ratio >= 0.90,
        "Uniform image should converge near-homogeneously; majority ratio = {:.4} (ones={}, zeros={})",
        ratio, ones, zeros
    );
}

// ── Test 3: Output is strictly binary ──────────────────────────────────────

#[test]
fn test_output_is_strictly_binary() {
    let dims = [10, 10, 10];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let ix = i % 10;
            if ix < 5 {
                20.0
            } else {
                180.0
            }
        })
        .collect();
    let image = make_image(data, dims);

    let result = ChanVeseSegmentation::new()
        .apply(&image)
        .expect("infallible: validated precondition");
    let vals = get_values(&result);

    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "Output voxel {} must be 0.0 or 1.0, got {}",
            i,
            v
        );
    }
}

// ── Test 4: Spatial metadata preserved ─────────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let origin = [1.5, -2.0, 3.7];
    let spacing = [0.5, 0.8, 1.2];
    let dims = [6, 6, 6];
    let n: usize = dims.iter().product();
    let data = vec![100.0_f32; n];
    let image = make_image_with_metadata(data, dims, origin, spacing);

    let result = ChanVeseSegmentation::new()
        .apply(&image)
        .expect("infallible: validated precondition");

    assert_eq!(result.origin(), image.origin(), "Origin must be preserved");
    assert_eq!(
        result.spacing(),
        image.spacing(),
        "Spacing must be preserved"
    );
    assert_eq!(
        result.direction(),
        image.direction(),
        "Direction must be preserved"
    );
    assert_eq!(result.shape(), dims, "Shape must be preserved");
}

// ── Test 5: Regularised Heaviside properties ───────────────────────────────

#[test]
fn test_regularised_heaviside_properties() {
    let eps = 1.0;
    let h0 = regularised_heaviside(0.0, eps);
    assert!(
        (h0 - 0.5).abs() < 1e-12,
        "H_ε(0) must equal 0.5, got {}",
        h0
    );
    let h_large = regularised_heaviside(1e6, eps);
    assert!(
        (h_large - 1.0).abs() < 1e-6,
        "H_ε(large) must approach 1.0, got {}",
        h_large
    );
    let h_neg = regularised_heaviside(-1e6, eps);
    assert!(
        h_neg.abs() < 1e-6,
        "H_ε(-large) must approach 0.0, got {}",
        h_neg
    );
    assert!(regularised_heaviside(-1.0, eps) < h0);
    assert!(h0 < regularised_heaviside(1.0, eps));
}

// ── Test 6: Regularised Dirac properties ───────────────────────────────────

#[test]
fn test_regularised_dirac_properties() {
    let eps = 1.0;
    let d0 = regularised_dirac(0.0, eps);
    let expected = 1.0 / (std::f64::consts::PI * eps);
    assert!(
        (d0 - expected).abs() < 1e-12,
        "δ_ε(0) must equal 1/(πε), got {} vs expected {}",
        d0,
        expected
    );
    let d_pos = regularised_dirac(2.5, eps);
    let d_neg = regularised_dirac(-2.5, eps);
    assert!(
        (d_pos - d_neg).abs() < 1e-15,
        "Dirac must be symmetric: {} vs {}",
        d_pos,
        d_neg
    );
    assert!(d0 > 0.0);
    assert!(d_pos > 0.0);
    assert!(d0 >= d_pos);
}

// ── Test 7: Curvature of a sphere ──────────────────────────────────────────

#[test]
fn test_curvature_of_sphere_phi() {
    let n = 11;
    let center = 5.0_f64;
    let radius = 3.5_f64;
    let total = n * n * n;
    let dims = [n, n, n];
    let mut phi = vec![0.0_f64; total];
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dz = iz as f64 - center;
                let dy = iy as f64 - center;
                let dx = ix as f64 - center;
                let r = (dz * dz + dy * dy + dx * dx).sqrt();
                phi[iz * n * n + iy * n + ix] = r - radius;
            }
        }
    }

    let mut kappa = vec![0.0_f64; total];
    compute_curvature(&phi, dims, &mut kappa);

    let test_idx = 5 * n * n + 5 * n + 8;
    assert!(
        kappa[test_idx] > 0.0,
        "Curvature of sphere φ at near-surface point must be positive, got {}",
        kappa[test_idx]
    );

    let center_idx = 5 * n * n + 5 * n + 5;
    assert!(
        kappa[center_idx].is_finite(),
        "Curvature at center must be finite, got {}",
        kappa[center_idx]
    );
}

// ── Test 8: Two-region slab with distinct intensities ──────────────────────

#[test]
fn test_two_region_slab() {
    let (nz, ny, nx) = (1, 10, 10);
    let n = nz * ny * nx;
    let mut data = vec![0.0_f32; n];
    for iy in 0..ny {
        for ix in 0..nx {
            data[iy * nx + ix] = if ix < 5 { 20.0 } else { 200.0 };
        }
    }

    let image = make_image(data.clone(), [nz, ny, nx]);
    let mut cv = ChanVeseSegmentation::new();
    cv.max_iterations = 500;
    let result = cv
        .apply(&image)
        .expect("infallible: validated precondition");
    let vals = get_values(&result);

    let mut left_ones = 0usize;
    let mut right_ones = 0usize;
    for iy in 0..ny {
        for ix in 0..nx {
            let v = vals[iy * nx + ix];
            if ix < 5 {
                if v == 1.0 {
                    left_ones += 1;
                }
            } else if v == 1.0 {
                right_ones += 1;
            }
        }
    }

    let left_majority_one = left_ones > 25;
    let right_majority_one = right_ones > 25;
    assert!(
        left_majority_one != right_majority_one,
        "Two-region slab should have distinct segmentation labels for each half (left_ones={}, right_ones={})",
        left_ones, right_ones
    );
}

// ── Test 9: Default construction ───────────────────────────────────────────

#[test]
fn test_default_matches_new() {
    let d = ChanVeseSegmentation::default();
    let n = ChanVeseSegmentation::new();
    assert_eq!(d.mu, n.mu);
    assert_eq!(d.nu, n.nu);
    assert_eq!(d.lambda1, n.lambda1);
    assert_eq!(d.lambda2, n.lambda2);
    assert_eq!(d.epsilon, n.epsilon);
    assert_eq!(d.dt, n.dt);
    assert_eq!(d.max_iterations, n.max_iterations);
    assert_eq!(d.tolerance, n.tolerance);
}
