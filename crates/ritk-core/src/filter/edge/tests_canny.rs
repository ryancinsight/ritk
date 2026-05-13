//! Tests for canny
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// A uniform image has zero gradient everywhere at interior voxels, so the
/// Canny detector must produce no edges in the interior.
///
/// **Proof**: ∇(constant) = 0 ⇒ |∇I| = 0 < T_low ⇒ no edges.
///
/// Note: The `GaussianFilter` uses burn's `conv1d` with zero-padding,
/// which introduces boundary artefacts on a nonzero constant image
/// (the convolution sees zeros outside the domain). Interior voxels
/// beyond the kernel support are unaffected.
#[test]
fn test_uniform_image_no_edges() {
    let dims = [24, 24, 24];
    let vals = vec![100.0_f32; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

    let detector = CannyEdgeDetector::new(1.0, 0.1, 0.2);
    let result = detector.apply(&img).unwrap();
    let out = extract_vals(&result);

    // Check only interior voxels: skip a margin of 6 voxels per side to
    // avoid the zero-padding boundary artefacts from the Gaussian
    // convolution (kernel support ≈ 3σ = 3, plus gradient stencil = 1,
    // plus safety = 2).
    let margin = 6;
    let [nz, ny, nx] = dims;
    let mut interior_edge_count = 0usize;
    for iz in margin..nz - margin {
        for iy in margin..ny - margin {
            for ix in margin..nx - margin {
                let flat = iz * ny * nx + iy * nx + ix;
                if out[flat] > 0.5 {
                    interior_edge_count += 1;
                }
            }
        }
    }
    assert!(
        interior_edge_count == 0,
        "uniform image should have no interior edges, but found {interior_edge_count} edge voxels"
    );
}

/// A step-edge image (half at 0.0, half at 100.0 along the x-axis) must
/// produce edges near the boundary plane.
///
/// **Derivation**: The gradient magnitude at the step is
/// |∇I| ≈ 100 / (2·sx) at the boundary voxels, which should exceed any
/// reasonable high threshold. After NMS, the boundary plane survives.
#[test]
fn test_step_edge_produces_edges() {
    let [nz, ny, nx] = [16usize, 16, 32];
    let n = nz * ny * nx;
    let half = nx / 2;
    let vals: Vec<f32> = (0..n)
        .map(|flat| {
            let ix = flat % nx;
            if ix < half {
                0.0
            } else {
                100.0
            }
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

    // Use a small sigma and thresholds that comfortably detect the 100-unit
    // step while ignoring noise.
    let detector = CannyEdgeDetector::new(1.0, 2.0, 10.0);
    let result = detector.apply(&img).unwrap();
    let out = extract_vals(&result);

    // There should be at least some nonzero voxels near the boundary
    let edge_count: usize = out.iter().filter(|&&v| v > 0.5).count();
    assert!(
        edge_count > 0,
        "step-edge image should produce edges, but edge_count = 0"
    );

    // Edges should be concentrated near the step (ix ≈ half).
    // The Gaussian with sigma=1.0 spreads the edge; use a wide margin
    // that accounts for the smoothing kernel support plus NMS/hysteresis
    // propagation.
    let margin = 8;
    let mut edges_near_step = 0usize;
    let mut edges_far_from_step = 0usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                if out[flat] > 0.5 {
                    if ix >= half.saturating_sub(margin) && ix <= half + margin {
                        edges_near_step += 1;
                    } else {
                        edges_far_from_step += 1;
                    }
                }
            }
        }
    }
    assert!(
        edges_near_step > edges_far_from_step,
        "edges should be concentrated near the step: near = {edges_near_step}, \
         far = {edges_far_from_step}"
    );
}

/// Low threshold must be ≤ high threshold.
#[test]
#[should_panic(expected = "low_threshold")]
fn test_invalid_thresholds() {
    let _ = CannyEdgeDetector::new(1.0, 5.0, 2.0);
}
