use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (vals, _) = ritk_tensor_ops::extract_vec(img).expect("infallible: validated precondition");
    vals
}

/// Circular signed-distance level set (φ < 0 inside), 2-D (`nz == 1`).
fn circle_phi(ny: usize, nx: usize, cy: f64, cx: f64, radius: f64) -> Vec<f32> {
    (0..ny * nx)
        .map(|i| {
            let iy = i / nx;
            let ix = i % nx;
            let d = ((iy as f64 - cy).powi(2) + (ix as f64 - cx).powi(2)).sqrt();
            (radius - d) as f32 // negate so φ < 0 inside is φ = -(d - r); here >0 inside
        })
        .collect()
}

/// A square feature with a circular initial contour evolves under the Canny
/// segmentation level set. The narrow-band solver must:
/// (1) keep φ finite everywhere,
/// (2) preserve the far-field magnitude exactly at `±(NumberOfLayers + 1) = ±3`,
/// (3) actually move the contour (non-trivial change near the zero crossing).
#[test]
fn test_canny_seg_level_set_structural() {
    let (ny, nx) = (30usize, 30);
    let mut feat = vec![0.0f32; ny * nx];
    for y in 8..22 {
        for x in 8..22 {
            feat[y * nx + x] = 100.0;
        }
    }
    // φ = (r − dist): >0 inside the circle. ITK convention is φ<0 inside, so use
    // the negation: φ = dist − r.
    let phi0: Vec<f32> = circle_phi(ny, nx, 15.0, 15.0, 4.0)
        .iter()
        .map(|&v| -v)
        .collect();

    let phi_img = make_image(phi0.clone(), [1, ny, nx]);
    let feat_img = make_image(feat, [1, ny, nx]);

    let filter = CannySegmentationLevelSet {
        canny_threshold: 10.0,
        canny_variance: 1.0,
        propagation_scaling: 1.0,
        curvature_scaling: 1.0,
        advection_scaling: 1.0,
        number_of_iterations: 5,
        max_rms_error: 0.0, // run all iterations
        iso_surface_value: 0.0,
    };

    let out = filter
        .apply(&phi_img, &feat_img)
        .expect("infallible: validated precondition");
    let result = extract_vals(&out);

    assert_eq!(result.len(), ny * nx, "output size must match input");
    assert!(
        result.iter().all(|v| v.is_finite()),
        "level set contains non-finite values"
    );
    // Far field is exactly ±(NL + 1) = ±3.
    assert!(
        result.iter().any(|&v| (v.abs() - 3.0).abs() < 1e-6),
        "far field must reach ±(NumberOfLayers + 1) = ±3"
    );
    // The narrow band is strictly within the far-field magnitude.
    assert!(
        result.iter().all(|&v| v.abs() <= 3.0 + 1e-6),
        "no value may exceed the far-field magnitude"
    );
    // The contour moved.
    let total_change: f32 = phi0
        .iter()
        .zip(result.iter())
        .map(|(&a, &b)| (b - a).abs())
        .sum();
    assert!(
        total_change > 1e-3,
        "level set should evolve: total_change={total_change:.4e}"
    );
}

/// Shape and finiteness preservation on a 3-D input over several iterations.
#[test]
fn test_canny_seg_level_set_3d_finite() {
    let [nz, ny, nx] = [12usize, 12, 12];
    let n = nz * ny * nx;
    let mut feat = vec![0.0f32; n];
    let mut phi0 = vec![0.0f32; n];
    let (cz, cy, cx) = (5.5, 5.5, 5.5);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let f = z * ny * nx + y * nx + x;
                if (3..9).contains(&z) && (3..9).contains(&y) && (3..9).contains(&x) {
                    feat[f] = 100.0;
                }
                let d =
                    ((z as f64 - cz).powi(2) + (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2))
                        .sqrt();
                phi0[f] = (d - 2.0) as f32; // φ<0 inside a radius-2 sphere
            }
        }
    }

    let phi_img = make_image(phi0, [nz, ny, nx]);
    let feat_img = make_image(feat, [nz, ny, nx]);

    let out = CannySegmentationLevelSet {
        canny_threshold: 10.0,
        canny_variance: 1.0,
        number_of_iterations: 5,
        max_rms_error: 0.0,
        ..Default::default()
    }
    .apply(&phi_img, &feat_img)
    .expect("infallible: validated precondition");
    let result = extract_vals(&out);

    assert_eq!(out.shape(), [nz, ny, nx]);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "3-D level set contains non-finite values"
    );
}
