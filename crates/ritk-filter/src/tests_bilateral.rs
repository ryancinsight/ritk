use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

/// Construct a test image from flat values and shape `[Z, Y, X]`.
fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

/// Extract flat `Vec<f32>` from an image (test utility).
fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
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

// ── 5. Equivalence vs brute-force reference ──────────────────────────

/// Reference implementation of the bilateral filter used only by this
/// test. It performs the inner loop with the original, completely
/// explicit arithmetic — no lookup table, no clamped iteration — so any
/// regression in `compute` (typo, sign flip, missing neighbour,
/// range/sigma swapped) will be caught by a value-level diff.
///
/// Tolerance: 1e-4 absolute. Both implementations accumulate the same
/// terms in the same floating-point order on this small volume, so
/// bitwise equality is also expected; we use a small absolute epsilon
/// to absorb unrelated LTO/optimisation variance if any.
#[test]
fn test_bilateral_matches_brute_force_reference() {
    let dims = [5usize, 6, 7];
    // Deterministic non-trivial data spanning a range that exercises
    // both spatial and intensity kernels.
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| 50.0 + ((i * 7 + 3) % 23) as f32).collect();

    let img = make_image(vals.clone(), dims);
    let filter = BilateralFilter::new(1.2, 4.0);
    let out = filter.apply(&img).unwrap();
    let actual = extract_vals(&out);

    // Brute-force reference (uses the original `compute` formula in plain
    // form, without our LUT / clamped-loop optimisation).
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let inv_two_ss2 = 1.0_f64 / (2.0 * 1.2_f64.powi(2));
    let inv_two_sr2 = 1.0_f64 / (2.0 * 4.0_f64.powi(2));

    let mut expected = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let c_flat = iz * ny * nx + iy * nx + ix;
                let c_val = vals[c_flat] as f64;

                let mut ws = 0.0_f64;
                let mut wt = 0.0_f64;
                let r = (3.0_f64 * 1.2).ceil() as isize;
                for dz in -r..=r {
                    let nz_i = iz as isize + dz;
                    if nz_i < 0 || nz_i >= nz as isize {
                        continue;
                    }
                    for dy in -r..=r {
                        let ny_i = iy as isize + dy;
                        if ny_i < 0 || ny_i >= ny as isize {
                            continue;
                        }
                        for dx in -r..=r {
                            let nx_i = ix as isize + dx;
                            if nx_i < 0 || nx_i >= nx as isize {
                                continue;
                            }
                            let n_flat =
                                nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                            let n_val = vals[n_flat] as f64;
                            let sd2 = (dz * dz + dy * dy + dx * dx) as f64;
                            let rd2 = (c_val - n_val) * (c_val - n_val);
                            let w = (-sd2 * inv_two_ss2 - rd2 * inv_two_sr2).exp();
                            ws += w * n_val;
                            wt += w;
                        }
                    }
                }
                expected[c_flat] = if wt > 1e-20 {
                    (ws / wt) as f32
                } else {
                    vals[c_flat]
                };
            }
        }
    }

    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-4,
            "voxel {i}: optimized={a} reference={e} (Δ={d:.3e})"
        );
    }
    assert!(
        max_abs < 1e-5,
        "max |a - e| = {max_abs:.3e} exceeds tight bound 1e-5 — possible floating-point reordering"
    );
}
