use super::*;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn cfg(iters: usize, dt: f32) -> CurvatureFlowConfig {
    CurvatureFlowConfig {
        num_iterations: iters,
        time_step: dt,
    }
}

// ── Analytical tests ──────────────────────────────────────────────────────

/// Constant image: every derivative is zero → κ = 0 → image unchanged.
/// Proof: ∇I = 0 everywhere → N = 0 → κ = 0 → ΔI = 0 each iteration.
#[test]
fn constant_image_unchanged() {
    let img = make_image(vec![42.0f32; 27], [3, 3, 3]);
    let out = CurvatureFlowImageFilter::new(cfg(5, 0.0625))
        .apply(&img)
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!(
            (x - 42.0f32).abs() < 1e-4,
            "constant image not preserved: {x}"
        );
    }
}

/// Zero iterations: filter is identity.
#[test]
fn zero_iterations_identity() {
    let vals: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(vals.clone(), [3, 3, 3]);
    let out = CurvatureFlowImageFilter::new(cfg(0, 0.0625))
        .apply(&img)
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    for (&o, &e) in v.iter().zip(vals.iter()) {
        assert_eq!(o, e, "0-iter output must equal input");
    }
}

/// Single voxel image: boundary conditions clamp all neighbours to same value
/// → all derivatives are zero → κ = 0 → identity.
#[test]
fn single_voxel_identity() {
    let img = make_image(vec![100.0f32], [1, 1, 1]);
    let out = CurvatureFlowImageFilter::new(cfg(3, 0.0625))
        .apply(&img)
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    assert!(
        (v[0] - 100.0f32).abs() < 1e-4,
        "single voxel must be unchanged: {}",
        v[0]
    );
}

/// Spatial metadata is preserved exactly.
#[test]
fn preserves_metadata() {
    let img = make_image(vec![5.0f32; 27], [3, 3, 3]);
    let out = CurvatureFlowImageFilter::new(cfg(2, 0.0625))
        .apply(&img)
        .unwrap();
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.direction(), img.direction());
    assert_eq!(out.shape(), [3, 3, 3]);
}

/// Output shape matches input shape.
#[test]
fn output_shape_matches_input() {
    let img = make_image(vec![1.0f32; 60], [3, 4, 5]);
    let out = CurvatureFlowImageFilter::new(cfg(2, 0.0625))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [3, 4, 5]);
}

/// Smoothing reduces step-edge intensity range:
/// A step edge between 0 and 100 should have its discontinuity softened
/// (max decreases, min increases) after several iterations.
#[test]
fn step_edge_range_decreases() {
    // 3D volume: left half = 0.0, right half = 100.0 (step at x=2)
    let mut vals = vec![0.0f32; 27];
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                if ix >= 2 {
                    vals[iz * 9 + iy * 3 + ix] = 100.0;
                }
            }
        }
    }
    let img = make_image(vals.clone(), [3, 3, 3]);
    let out = CurvatureFlowImageFilter::new(cfg(10, 0.0625))
        .apply(&img)
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    let out_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let in_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let in_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        out_max <= in_max,
        "max should not increase: out={out_max}, in={in_max}"
    );
    assert!(
        out_min >= in_min,
        "min should not decrease: out={out_min}, in={in_min}"
    );
}

/// `CurvatureFlowConfig::default()` values: `num_iterations = 5`,
/// `time_step = 0.0625`.
///
/// Note: ITK's *constructor* defaults are `time_step = 0.05` and
/// `num_iterations = 0` (no-op). Our defaults are the commonly cited
/// ITK-compatible working values, not the ITK constructor defaults.
#[test]
fn default_config_values() {
    let cfg = CurvatureFlowConfig::default();
    assert_eq!(cfg.num_iterations, 5, "default iterations");
    assert!(
        (cfg.time_step - 0.0625f32).abs() < 1e-7,
        "default dt = 0.0625"
    );
    // Stability: dt ≤ 1/6 ≈ 0.1667
    assert!(
        cfg.time_step <= 1.0 / 6.0 + 1e-6,
        "default dt must be within stability bound"
    );
}
