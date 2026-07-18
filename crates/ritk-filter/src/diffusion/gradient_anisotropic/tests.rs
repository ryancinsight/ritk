use super::*;

use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_spatial::{Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

// T1 â€” Constant image: update = 0 everywhere â†’ output = input exactly.
//
// Proof: all differences delta = I(q) - I(p) = c - c = 0 for all
// neighbours q, so conductance_exp(0, K) Â· 0 = 0 for every term.
// Thus update = 0 and I_new = I for every iteration.
#[test]
fn constant_image_is_identity() {
    let v = 42.0_f32;
    let dims = [4, 4, 4];
    let img = make_image(vec![v; dims[0] * dims[1] * dims[2]], dims);
    let cfg = GradientDiffusionConfig {
        num_iterations: 10,
        time_step: 0.125,
        conductance: 1.0,
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    for &r in result {
        assert!((r - v).abs() < 1e-5, "expected {v}, got {r}");
    }
}

// T2 â€” Zero iterations: output = input exactly.
//
// Proof: num_iterations = 0 â†’ the iteration loop does not execute â†’
// cur is returned unchanged.
#[test]
fn zero_iterations_is_identity() {
    let vals: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(vals.clone(), [3, 3, 3]);
    let cfg = GradientDiffusionConfig {
        num_iterations: 0,
        time_step: 0.125,
        conductance: 1.0,
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    for (i, (&r, &v)) in result.iter().zip(vals.iter()).enumerate() {
        assert!((r - v).abs() < 1e-6, "voxel {i}: expected {v}, got {r}");
    }
}

// T3 â€” Large conductance K â†’ isotropic Laplacian smoothing.
//
// When K â†’ âˆž, c(s) = exp(-(s/K)Â²) â†’ 1 for all finite s.
// A step edge of 100.0 HU with K = 1e6 embeds the boundary at z=4/z=5.
// After 1 iteration with Î”t=0.125, voxel 4 (v=0) gains from its z=5
// neighbour (100): update â‰ˆ 0 + 0.125Â·100 = 12.5 â†’ out[4] = 12.5 > 0.
// Thus min(output) > 0.0 after one iteration, proving diffusion occurred.
// (max may stay 100 at the far end; only the interior boundary proves motion.)
#[test]
fn large_k_produces_isotropic_smoothing() {
    // 1-D step edge along z-axis embedded in 10Ã—1Ã—1.
    // First 5 voxels = 0.0, last 5 = 100.0.
    let mut vals = vec![0.0_f32; 10];
    for v in &mut vals[5..] {
        *v = 100.0;
    }
    let img = make_image(vals, [10, 1, 1]);
    let cfg = GradientDiffusionConfig {
        num_iterations: 1,
        time_step: 0.125,
        conductance: 1e6, // c(s) â‰ˆ 1 for all finite s
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    // Voxel 4 (boundary, v=0) must have gained from its z=5 neighbour (v=100).
    // Analytical: out[4] = 0 + 0.125 Â· c(100, 1e6) Â· 100
    //           = 0 + 0.125 Â· exp(-(100/1e6)Â²) Â· 100
    //           â‰ˆ 0 + 0.125 Â· 1.0 Â· 100 = 12.5
    assert!(
        result[4] > 1.0,
        "voxel 4 should gain from z=5 neighbour; got {}",
        result[4]
    );
    // Voxel 5 (boundary, v=100) must have lost to its z=4 neighbour (v=0).
    // Analytical: out[5] = 100 + 0.125 Â· c(100, 1e6) Â· (0-100) â‰ˆ 87.5
    assert!(
        result[5] < 99.0,
        "voxel 5 should lose to z=4 neighbour; got {}",
        result[5]
    );
}

// T4 â€” Small conductance K â†’ edge preservation.
//
// When K â†’ 0, c(s) = exp(-(s/K)Â²) â†’ 0 for all s â‰  0.
// A step edge of 100.0 HU with K = 0.001 produces c(100/0.001) â‰ˆ 0.
// After 5 iterations, the step edge must be substantially preserved:
// the original min=0, max=100 values must not collapse toward each other.
//
// Analytical bound: max(output) > 99.0 and min(output) < 1.0.
#[test]
fn small_k_preserves_edges() {
    let mut vals = vec![0.0_f32; 10];
    for v in &mut vals[5..] {
        *v = 100.0;
    }
    let img = make_image(vals, [10, 1, 1]);
    let cfg = GradientDiffusionConfig {
        num_iterations: 5,
        time_step: 0.125,
        conductance: 0.001, // c(100/0.001) â‰ˆ 0
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    let max_out = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_out = result.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(max_out > 99.0, "small K should preserve max; got {max_out}");
    assert!(min_out < 1.0, "small K should preserve min; got {min_out}");
}

// T5 â€” Single-voxel image (1Ã—1Ã—1): no neighbours â†’ update = 0.
//
// With no neighbours in the 6-neighbourhood (all boundary conditions), the
// update sum is 0 for every iteration.  Output = input exactly.
#[test]
fn single_voxel_is_identity() {
    let v = 77.0_f32;
    let img = make_image(vec![v], [1, 1, 1]);
    let cfg = GradientDiffusionConfig::default();
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    assert!(
        (result[0] - v).abs() < 1e-6,
        "expected {v}, got {}",
        result[0]
    );
}

// T6 â€” Spatial metadata preserved through filter application.
//
// Origin, spacing, and direction of the output image must equal those of
// the input; the filter must not modify spatial metadata.
#[test]
fn spatial_metadata_preserved() {
    use ritk_spatial::Direction;
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 1.0]);
    let direction = Direction::identity();
    let tensor = Tensor::<f32, B>::from_slice([2, 2, 2], &[1.0_f32; 8]);
    let img = Image::new(tensor, origin, spacing, direction);
    let cfg = GradientDiffusionConfig::default();
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}

// T8 â€” One iteration: analytical update for interior voxel of step edge.
//
// Image: 1Ã—1Ã—3, values [0.0, 50.0, 100.0].
// Middle voxel p = (0,0,1), v = 50.0.
// Neighbours: (0,0,0)=0.0 â†’ delta=-50, (0,0,2)=100.0 â†’ delta=+50.
// K = 100.0 (large, c â‰ˆ 1 for delta=50): c(50/100) = exp(-(0.5)Â²) = exp(-0.25).
// update = exp(-0.25)Â·(-50) + exp(-0.25)Â·50 = 0 (symmetric cancellation).
// I_new(1) = 50 + Î”tÂ·0 = 50.0 exactly.
#[test]
fn symmetric_step_middle_voxel_unchanged() {
    let img = make_image(vec![0.0, 50.0, 100.0], [3, 1, 1]);
    let cfg = GradientDiffusionConfig {
        num_iterations: 1,
        time_step: 0.125,
        conductance: 100.0,
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    // Middle voxel must be unchanged by symmetry.
    assert!(
        (result[1] - 50.0).abs() < 1e-4,
        "expected 50.0, got {}",
        result[1]
    );
}

// T9 â€” Diffusion reduces gradient magnitude over multiple iterations.
//
// A sharp 2-voxel step edge in a 1Ã—1Ã—10 image must have its peak
// finite-difference |I[i+1]-I[i]| reduced after 10 iterations with
// moderate conductance K=50.  This verifies that the filter performs
// genuine smoothing (non-identity) for non-trivial input.
#[test]
fn diffusion_reduces_gradient_magnitude() {
    let mut vals = vec![0.0_f32; 10];
    for v in &mut vals[5..] {
        *v = 100.0;
    }
    let peak_before: f32 = vals
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);

    let img = make_image(vals, [10, 1, 1]);
    let cfg = GradientDiffusionConfig {
        num_iterations: 10,
        time_step: 0.125,
        conductance: 50.0,
    };
    let out = GradientAnisotropicDiffusionFilter::new(cfg)
        .apply::<B>(&img)
        .unwrap();
    let result = out.data().as_slice();
    let peak_after: f32 = result
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        peak_after < peak_before,
        "peak gradient should decrease; before={peak_before}, after={peak_after}"
    );
}
