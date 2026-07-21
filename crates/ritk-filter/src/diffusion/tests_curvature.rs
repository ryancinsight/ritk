use super::*;
use coeus_core::SequentialBackend;
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn image_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

// ── Test 1: constant image must be unchanged ───────────────────────────────

#[test]
fn test_constant_image_unchanged() {
    let dims = [8, 8, 8];
    let vals = vec![5.0f32; 8 * 8 * 8];
    let img = make_image(vals.clone(), dims);

    let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig::default());
    let out = filter.apply(&img).expect("infallible: validated precondition");
    let out_vals = image_vals(&out);

    let max_diff = out_vals
        .iter()
        .zip(vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "constant image must be unchanged; max diff = {max_diff}"
    );
}

#[test]
fn native_curvature_preserves_geometry_and_matches_kernel() {
    let dimensions = [2, 3, 4];
    let values: Vec<f32> = (0..24).map(|index| index as f32 * 0.25).collect();
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let config = CurvatureConfig {
        num_iterations: 2,
        time_step: 1.0 / 16.0,
        conductance: 3.0,
    };

    let output = CurvatureAnisotropicDiffusionFilter::new(config.clone())
        .apply_native(&image, &SequentialBackend)
        .expect("native curvature succeeds");

    assert_eq!(output.shape(), dimensions);
    assert_eq!(*output.origin(), origin);
    assert_eq!(*output.spacing(), spacing);
    assert_eq!(*output.direction(), direction);
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        curvature_diffuse(&values, dimensions, [0.5, 1.0, 2.0], &config)
    );
}

// ── Test 2: linear field — deep interior must be unchanged ─────────────────
// A linear ramp has zero curvature, so the MCDE speed is identically 0 wherever
// the stencil sees only real (unclamped) data. ITK's ZeroFluxNeumann boundary
// does perturb the ramp at the edges (the boundary acts as a reflecting wall),
// and that perturbation propagates one voxel inward per iteration. Voxels at
// Chebyshev distance > num_iterations from every boundary are therefore exactly
// unchanged.

#[test]
fn test_linear_field_deep_interior_unchanged() {
    let [nz, ny, nx] = [30usize, 30, 30];
    let iters = 5usize;
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|i| {
            let ix = (i % nx) as f32;
            let iy = ((i / nx) % ny) as f32;
            let iz = (i / (ny * nx)) as f32;
            0.3 * ix + 0.5 * iy + 0.7 * iz
        })
        .collect();
    let img = make_image(vals.clone(), [nz, ny, nx]);

    let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
        num_iterations: iters,
        time_step: 1.0 / 16.0,
        conductance: 3.0,
    });
    let out = filter.apply(&img).expect("infallible: validated precondition");
    let out_vals = image_vals(&out);

    // Margin of (iters + 1) keeps every stencil access away from the propagated
    // boundary perturbation.
    let m = iters + 1;
    let mut max_diff = 0.0f32;
    for iz in m..nz - m {
        for iy in m..ny - m {
            for ix in m..nx - m {
                let i = iz * ny * nx + iy * nx + ix;
                max_diff = max_diff.max((out_vals[i] - vals[i]).abs());
            }
        }
    }
    assert!(
        max_diff < 1e-4,
        "linear field deep interior must be unchanged; max diff = {max_diff}"
    );
}

// ── Test 3: mean conservation ──────────────────────────────────────────────

#[test]
fn test_mean_conservation() {
    let dims = [12, 12, 12];
    let n = 12 * 12 * 12;
    // Sinusoidal image with non-trivial curvature.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let ix = (i % 12) as f32 / 12.0;
            let iy = ((i / 12) % 12) as f32 / 12.0;
            let iz = (i / 144) as f32 / 12.0;
            (std::f32::consts::PI * ix).sin()
                * (std::f32::consts::PI * iy).cos()
                * (std::f32::consts::PI * iz).sin()
                + 5.0
        })
        .collect();
    let mean_in: f32 = vals.iter().sum::<f32>() / n as f32;

    let img = make_image(vals, dims);
    let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
        num_iterations: 20,
        time_step: 1.0 / 16.0,
        conductance: 3.0,
    });
    let out = filter.apply(&img).expect("infallible: validated precondition");
    let out_vals = image_vals(&out);
    let mean_out: f32 = out_vals.iter().sum::<f32>() / n as f32;

    let rel_err = ((mean_out - mean_in) / mean_in).abs();
    assert!(
        rel_err < 1e-2,
        "mean should be approximately conserved; rel err = {rel_err}"
    );
}

// ── Test 4: spherical blob smoothed (gradient reduced) ────────────────────

#[test]
fn test_spherical_blob_smoothed() {
    let [nz, ny, nx] = [24usize, 24, 24];
    let n = nz * ny * nx;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let ix = (i % nx) as f32 - 12.0;
            let iy = ((i / nx) % ny) as f32 - 12.0;
            let iz = (i / (ny * nx)) as f32 - 12.0;
            let r = (ix * ix + iy * iy + iz * iz).sqrt();
            if r < 6.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // Compute max gradient magnitude before filtering.
    let max_grad_in = max_gradient_magnitude(&vals, [nz, ny, nx]);

    let img = make_image(vals, [nz, ny, nx]);
    let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
        num_iterations: 10,
        time_step: 1.0 / 16.0,
        conductance: 3.0,
    });
    let out = filter.apply(&img).expect("infallible: validated precondition");
    let out_vals = image_vals(&out);
    let max_grad_out = max_gradient_magnitude(&out_vals, [nz, ny, nx]);

    assert!(
            max_grad_out < max_grad_in,
            "spherical blob should be smoothed; max grad before={max_grad_in:.4} after={max_grad_out:.4}"
        );
}

// ── Test 5: stability — outputs finite and within intensity range ──────────

#[test]
fn test_stability_small_timestep() {
    let [nz, ny, nx] = [10usize, 10, 10];
    let n = nz * ny * nx;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let ix = i % nx;
            let iy = (i / nx) % ny;
            let iz = i / (ny * nx);
            ((ix + iy * 3 + iz * 7) % 17) as f32 * 10.0
        })
        .collect();
    let v_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let img = make_image(vals, [nz, ny, nx]);
    let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
        num_iterations: 50,
        time_step: 1.0 / 16.0,
        conductance: 3.0,
    });
    let out = filter.apply(&img).expect("infallible: validated precondition");
    let out_vals = image_vals(&out);

    for &v in &out_vals {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
    // Allow a small overshoot margin due to finite-difference approximation.
    let margin = (v_max - v_min) * 0.05;
    for &v in &out_vals {
        assert!(
            v >= v_min - margin && v <= v_max + margin,
            "output value {v} outside input range [{}, {}] (+margin)",
            v_min - margin,
            v_max + margin
        );
    }
}

// ── Helper: max gradient magnitude via central differences ─────────────────

fn max_gradient_magnitude(data: &[f32], dims: [usize; 3]) -> f32 {
    let [nz, ny, nx] = dims;
    let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
    let mut max_g = 0.0f32;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let gz = (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) * 0.5;
                let gy = (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) * 0.5;
                let gx = (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) * 0.5;
                let g = (gz * gz + gy * gy + gx * gx).sqrt();
                max_g = max_g.max(g);
            }
        }
    }
    max_g
}
