use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (vals, _) = ritk_tensor_ops::extract_vec(img).unwrap();
    vals
}

/// Output is the binary segmentation: every voxel is exactly 0.0 or 1.0, the
/// shape is preserved, and the bright feature region is captured (φ < 0 → 1).
#[test]
fn test_chan_and_vese_binary_output_captures_feature() {
    let (ny, nx) = (24usize, 24);
    let mut feat = vec![0.0f32; ny * nx];
    for y in 6..18 {
        for x in 6..18 {
            feat[y * nx + x] = 100.0;
        }
    }
    // φ₀ = dist − 7: negative inside a radius-7 circle (covers the bright square).
    let phi0: Vec<f32> = (0..ny * nx)
        .map(|i| {
            let (iy, ix) = (i / nx, i % nx);
            (((iy as f64 - 12.0).powi(2) + (ix as f64 - 12.0).powi(2)).sqrt() - 7.0) as f32
        })
        .collect();

    let out = ScalarChanAndVeseDenseLevelSet {
        number_of_iterations: 5,
        ..Default::default()
    }
    .apply(
        &make_image(phi0, [1, ny, nx]),
        &make_image(feat.clone(), [1, ny, nx]),
    )
    .unwrap();
    let result = extract_vals(&out);

    assert_eq!(result.len(), ny * nx);
    assert!(
        result.iter().all(|&v| v == 0.0 || v == 1.0),
        "output must be a binary segmentation"
    );
    // The bright-square centre must be labelled foreground.
    assert_eq!(result[12 * nx + 12], 1.0, "feature centre must be inside");
    // A far background corner must be labelled background.
    assert_eq!(result[0], 0.0, "corner must be outside");
    // The segmented region should overlap the bright square substantially.
    let seg_in_square: usize = (6..18)
        .flat_map(|y| (6..18).map(move |x| (y, x)))
        .filter(|&(y, x)| result[y * nx + x] == 1.0)
        .count();
    assert!(
        seg_in_square >= 90,
        "segmentation must capture most of the bright square, got {seg_in_square}"
    );
}

/// 3-D stability: the binary output stays well-formed and finite over several
/// iterations with per-iteration Maurer reinitialization.
#[test]
fn test_chan_and_vese_3d_binary_stable() {
    let [nz, ny, nx] = [12usize, 12, 12];
    let n = nz * ny * nx;
    let (cz, cy, cx) = (5.5, 5.5, 5.5);
    let mut feat = vec![0.0f32; n];
    let mut phi0 = vec![0.0f32; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let f = z * ny * nx + y * nx + x;
                let d =
                    ((z as f64 - cz).powi(2) + (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2))
                        .sqrt();
                if d <= 3.0 {
                    feat[f] = 100.0;
                }
                phi0[f] = (d - 4.0) as f32;
            }
        }
    }

    let out = ScalarChanAndVeseDenseLevelSet {
        number_of_iterations: 4,
        ..Default::default()
    }
    .apply(
        &make_image(phi0, [nz, ny, nx]),
        &make_image(feat, [nz, ny, nx]),
    )
    .unwrap();
    let result = extract_vals(&out);

    assert_eq!(out.shape(), [nz, ny, nx]);
    assert!(
        result.iter().all(|&v| v == 0.0 || v == 1.0),
        "binary output"
    );
    assert!(result.contains(&1.0), "must segment some foreground");
    assert!(result.contains(&0.0), "must keep some background");
}
