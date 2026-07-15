//! Tests for canny
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = LegacyBurnBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(vals, dims, spacing)
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

    let detector = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 0.1, 0.2);
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
    let detector = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 2.0, 10.0);
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
    let _ = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 5.0, 2.0);
}

/// A 2-D step edge (20×20×1, vertical boundary at x=10) must produce edge
/// pixels concentrated at the step after sub-pixel NMS and hysteresis.
///
/// **Derivation**: The gradient magnitude at the step is
/// |\u2207I| ≈ 1 / (2·sx) per normalised unit step, which greatly exceeds the
/// high threshold of 0.15. Sub-pixel NMS retains the single-voxel-wide ridge;
/// hysteresis BFS connects the whole column.
#[test]
fn test_canny_2d_step_edge_pixel_count() {
    let ny = 20usize;
    let nx = 20usize;
    let nz = 1usize;
    let mut data = vec![0.0f32; nz * ny * nx];
    for iy in 0..ny {
        for ix in 10..nx {
            data[iy * nx + ix] = 1.0;
        }
    }
    let img = make_image(data, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let detector = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 0.05, 0.15);
    let result = detector.apply(&img).unwrap();
    let out = extract_vals(&result);
    // Count rows where either x=9 or x=10 is an edge pixel.
    let edge_count: usize = (0..ny)
        .filter(|&iy| out[iy * nx + 9] > 0.5 || out[iy * nx + 10] > 0.5)
        .count();
    assert!(
        edge_count >= 15,
        "Expected >= 15 edge pixels at step, got {edge_count}"
    );
}

/// A linear ramp image has a spatially uniform gradient, so after NMS the
/// surviving voxels form a single-pixel-wide ridge (at most one per row).
///
/// **Derivation**: NMS keeps only local maxima along the gradient direction;
/// a uniform gradient field has at most one local maximum per gradient line,
/// so fewer than 30 % of voxels should survive.
#[test]
fn test_canny_nms_reduces_thick_edges() {
    let ny = 20usize;
    let nx = 20usize;
    let nz = 1usize;
    let data: Vec<f32> = (0..ny * nx).map(|i| (i % nx) as f32 / nx as f32).collect();
    let img = make_image(data, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let detector = CannyEdgeDetector::new(GaussianSigma::new_unchecked(0.5), 0.03, 0.06);
    let result = detector.apply(&img).unwrap();
    let out = extract_vals(&result);
    let edge_count = out.iter().filter(|&&v| v > 0.5).count();
    assert!(
        edge_count < (ny * nx * 30) / 100,
        "Too many edge pixels (NMS not thinning): {edge_count}"
    );
}
