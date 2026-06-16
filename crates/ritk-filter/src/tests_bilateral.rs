use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

/// Construct a test image from flat values and shape `[Z, Y, X]`.
fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

/// Extract flat `Vec<f32>` from an image (test utility).
fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

// ── 1. Uniform image → unchanged ─────────────────────────────────────

/// A constant image has zero range differences everywhere, so the
/// bilateral filter reduces to a spatial Gaussian average of identical
/// values.  Output must equal the constant.
#[test]
fn test_bilateral_uniform_image_unchanged() {
    let dims = [6, 8, 10];
    let val = 7.5_f32;
    let vals = vec![val; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims);

    let filter = BilateralFilter::new(1.5, 10.0);
    let out = filter.apply(&img).unwrap();

    let result = extract_vals(&out);
    assert_eq!(result.len(), dims[0] * dims[1] * dims[2]);
    for (i, &v) in result.iter().enumerate() {
        assert!((v - val).abs() < 1e-5, "voxel {i}: expected {val}, got {v}");
    }
}

// ── 2. Edge preservation ─────────────────────────────────────────────

/// Step edge along the X axis: left half = 20, right half = 200.
/// With a tight range sigma the bilateral filter should NOT blur across
/// the edge.  Voxels far from the boundary must remain near their
/// original value.
#[test]
fn test_bilateral_edge_preservation() {
    let [nz, ny, nx] = [8usize, 8, 16];
    let n = nz * ny * nx;
    let mut vals = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                vals[iz * ny * nx + iy * nx + ix] = if ix < nx / 2 { 20.0 } else { 200.0 };
            }
        }
    }
    let img = make_image(vals, [nz, ny, nx]);

    // Tight range sigma → intensity difference across edge (180) ≫ σ_r
    // so cross-edge weights are negligible.
    let filter = BilateralFilter::new(1.0, 5.0);
    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    // Check voxels well inside each region (≥2 voxels from boundary).
    for iz in 0..nz {
        for iy in 0..ny {
            // Left interior.
            for ix in 0..(nx / 2 - 2) {
                let v = result[iz * ny * nx + iy * nx + ix];
                assert!(
                    (v - 20.0).abs() < 2.0,
                    "left interior voxel [{iz},{iy},{ix}]: expected ~20, got {v}"
                );
            }
            // Right interior.
            for ix in (nx / 2 + 2)..nx {
                let v = result[iz * ny * nx + iy * nx + ix];
                assert!(
                    (v - 200.0).abs() < 2.0,
                    "right interior voxel [{iz},{iy},{ix}]: expected ~200, got {v}"
                );
            }
        }
    }
}

// ── 3. Metadata preserved ────────────────────────────────────────────

/// Origin, spacing, and direction of the output image must match the
/// input exactly.
#[test]
fn test_bilateral_metadata_preserved() {
    let dims = [4, 4, 4];
    let vals = vec![1.0_f32; 64];
    let img = make_image(vals, dims);

    let filter = BilateralFilter::new(1.0, 1.0);
    let out = filter.apply(&img).unwrap();

    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.direction(), img.direction());
    assert_eq!(out.shape(), img.shape());
}

// ── 4. Smooth region is smoothed ─────────────────────────────────────

/// In a uniform region with additive noise, the bilateral filter should
/// reduce variance while keeping the mean approximately unchanged.
///
/// Construction: 8×8×8 image with base value 100.  A deterministic
/// noise pattern (±5 alternating) is added.  After bilateral filtering
/// with a large range sigma (noise amplitude ≪ σ_r), variance must
/// decrease.
#[test]
fn test_bilateral_smooth_region_is_smoothed() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let n = nz * ny * nx;
    let base = 100.0_f32;
    let noise_amp = 5.0_f32;

    // Deterministic alternating noise: +5 / -5 in a checkerboard.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let iz = i / (ny * nx);
            let iy = (i / nx) % ny;
            let ix = i % nx;
            let sign = if (iz + iy + ix) % 2 == 0 { 1.0 } else { -1.0 };
            base + noise_amp * sign
        })
        .collect();

    let input_mean = vals.iter().sum::<f32>() / n as f32;
    let input_var = vals
        .iter()
        .map(|&v| (v - input_mean) * (v - input_mean))
        .sum::<f32>()
        / n as f32;

    let img = make_image(vals, [nz, ny, nx]);

    // Large range sigma so noise is within the range kernel → smoothed.
    let filter = BilateralFilter::new(1.5, 50.0);
    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    let output_mean = result.iter().sum::<f32>() / n as f32;
    let output_var = result
        .iter()
        .map(|&v| (v - output_mean) * (v - output_mean))
        .sum::<f32>()
        / n as f32;

    // Mean should be approximately conserved.
    assert!(
        (output_mean - input_mean).abs() < 1.0,
        "mean shifted: input={input_mean:.4} output={output_mean:.4}"
    );

    // Variance must decrease.
    assert!(
        output_var < input_var,
        "variance not reduced: input={input_var:.4} output={output_var:.4}"
    );

    // Quantitative check: variance should drop substantially.
    assert!(
        output_var < input_var * 0.5,
        "variance reduction insufficient: input={input_var:.4} output={output_var:.4}"
    );
}
