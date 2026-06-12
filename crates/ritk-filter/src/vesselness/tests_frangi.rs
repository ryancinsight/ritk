//! Tests for frangi
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_tensor_ops::extract_vec_infallible;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── gaussian_blur_vec ─────────────────────────────────────────────────────

/// Blurring a uniform image must return the same uniform image.
#[test]
fn test_gaussian_blur_uniform_invariant() {
    let dims = [8usize, 8, 8];
    let data = vec![5.0f32; 8 * 8 * 8];
    let blurred = gaussian_blur_vec(&data, dims, 1.0, [1.0, 1.0, 1.0]);
    for (i, &v) in blurred.iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1e-4,
            "uniform blur: voxel {i} expected 5.0, got {v}"
        );
    }
}

/// Gaussian blur must reduce peak intensity (smoothing spreads energy).
#[test]
fn test_gaussian_blur_smooths_peak() {
    let dims = [9usize, 9, 9];
    let n = 9 * 9 * 9;
    let mut data = vec![0.0f32; n];
    // Single bright voxel at centre.
    data[4 * 9 * 9 + 4 * 9 + 4] = 1000.0;
    let blurred = gaussian_blur_vec(&data, dims, 1.5, [1.0, 1.0, 1.0]);
    let peak = blurred.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(peak < 1000.0, "blur must reduce peak; peak = {peak}");
    // Energy (sum) must be conserved under replicate boundary.
    let sum_orig: f32 = data.iter().sum();
    let sum_blur: f32 = blurred.iter().sum();
    // Sum is not necessarily preserved under clamp-to-edge blurring when
    // the image is mostly zero.  Check only that no energy is generated.
    assert!(
        sum_blur <= sum_orig * 1.01,
        "blur must not create energy: orig={sum_orig}, blurred={sum_blur}"
    );
}

// ── FrangiVesselnessFilter ────────────────────────────────────────────────

/// **Test 1 — Cylindrical tube phantom.**
///
/// 20×20×20 image.  All voxels whose cross-sectional distance from the
/// z-axis centre (y = 9.5, x = 9.5) is < 3 voxels are set to 100.0;
/// the background is 0.0.
///
/// Invariants verified:
/// - Tube centre (iz ∈ {9,10}, iy ∈ {9,10}, ix ∈ {9,10}): V > 0.05.
/// - Corners (0,0,0) and (19,19,19): V < 0.02.
#[test]
fn test_frangi_cylindrical_tube() {
    const N: usize = 20;
    let mut vals = vec![0.0f32; N * N * N];
    let centre = (N as f64 - 1.0) / 2.0; // 9.5
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let dy = iy as f64 - centre;
                let dx = ix as f64 - centre;
                if (dy * dy + dx * dx).sqrt() < 3.0 {
                    vals[iz * N * N + iy * N + ix] = 100.0;
                }
            }
        }
    }

    let image = make_image(vals, [N, N, N]);
    let config = FrangiConfig {
        scales: vec![1.0, 2.0],
        alpha: 0.5,
        beta: 0.5,
        gamma: 15.0,
        polarity: VesselPolarity::Bright,
    };
    let filter = FrangiVesselnessFilter::new(config);
    let out = filter.apply(&image).expect("frangi apply failed");

    let (v, _) = extract_vec_infallible(&out);

    let get = |iz: usize, iy: usize, ix: usize| v[iz * N * N + iy * N + ix];

    // Tube-centre voxels: vesselness must be clearly positive.
    for iz in [9usize, 10] {
        for iy in [9usize, 10] {
            for ix in [9usize, 10] {
                let val = get(iz, iy, ix);
                assert!(
                    val > 0.05,
                    "tube centre ({iz},{iy},{ix}): expected > 0.05, got {val}"
                );
            }
        }
    }

    // Corner voxels: far from tube, vesselness must be near zero.
    for (iz, iy, ix) in [(0usize, 0usize, 0usize), (N - 1, N - 1, N - 1)] {
        let val = get(iz, iy, ix);
        assert!(
            val < 0.02,
            "corner ({iz},{iy},{ix}): expected < 0.02, got {val}"
        );
    }
}

/// **Test 2 — Uniform image.**
///
/// All second derivatives are zero at every voxel → S = 0 → V = 0.
#[test]
fn test_frangi_uniform_image_zero_vesselness() {
    const N: usize = 10;
    let vals = vec![42.0f32; N * N * N];
    let image = make_image(vals, [N, N, N]);
    let filter = FrangiVesselnessFilter::new(FrangiConfig::default());
    let out = filter.apply(&image).expect("frangi apply failed");

    let (v, _) = extract_vec_infallible(&out);
    for (i, &val) in v.iter().enumerate() {
        assert!(val < 1e-6, "uniform image: voxel {i} expected 0, got {val}");
    }
}

/// **Test 3 — Spherical blob.**
///
/// A bright sphere of radius 5 centred in a 30×30×30 image is a
/// blob-like structure, not a vessel.  With `bright_vessels = true` and
/// the Frangi measure, a perfect sphere satisfies the polarity gate
/// (both λ₂ and λ₃ are negative) but the blobness term
/// `exp(−R_B² / (2β²))` with R_B ≈ 1 and β = 0.5 evaluates to ≈ 0.135,
/// yielding a vesselness value substantially lower than that of a tube
/// (where R_B ≈ 0 → blobness term ≈ 1.0).
///
/// Invariant: sphere centre vesselness < tube scale factor × 0.3.
/// Concretely: V_sphere_centre < 0.4.
#[test]
fn test_frangi_sphere_low_vesselness() {
    const N: usize = 30;
    let mut vals = vec![0.0f32; N * N * N];
    let centre = (N as f64 - 1.0) / 2.0; // 14.5
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let dz = iz as f64 - centre;
                let dy = iy as f64 - centre;
                let dx = ix as f64 - centre;
                if (dz * dz + dy * dy + dx * dx).sqrt() < 5.0 {
                    vals[iz * N * N + iy * N + ix] = 100.0;
                }
            }
        }
    }

    let image = make_image(vals, [N, N, N]);
    let config = FrangiConfig {
        scales: vec![1.0, 2.0],
        alpha: 0.5,
        beta: 0.5,
        gamma: 15.0,
        polarity: VesselPolarity::Bright,
    };
    let filter = FrangiVesselnessFilter::new(config);
    let out = filter.apply(&image).expect("frangi apply failed");

    let (v, _) = extract_vec_infallible(&out);

    // Sphere-centre voxel.
    let c = 14usize; // floor(14.5)
    let centre_idx = c * N * N + c * N + c;
    let val = v[centre_idx];

    // A sphere has R_B ≈ 1 → blobness term ≈ exp(-2) ≈ 0.135, so
    // vesselness is suppressed relative to a tube.  Threshold: < 0.4.
    assert!(
        val < 0.4,
        "sphere centre vesselness: expected < 0.4 (blob suppression), got {val}"
    );
}

/// **Test 4 — Tube vs. sphere discrimination.**
///
/// The vesselness at the tube centre must exceed the vesselness at the
/// sphere centre.  This directly validates that the Frangi measure
/// discriminates tubular from blob-like structures.
#[test]
fn test_frangi_tube_exceeds_sphere() {
    const N: usize = 20;
    // ── Tube phantom ──────────────────────────────────────────────────────
    let mut tube_vals = vec![0.0f32; N * N * N];
    let centre = (N as f64 - 1.0) / 2.0;
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let dy = iy as f64 - centre;
                let dx = ix as f64 - centre;
                if (dy * dy + dx * dx).sqrt() < 3.0 {
                    tube_vals[iz * N * N + iy * N + ix] = 100.0;
                }
            }
        }
    }

    // ── Sphere phantom ────────────────────────────────────────────────────
    let mut sphere_vals = vec![0.0f32; N * N * N];
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let dz = iz as f64 - centre;
                let dy = iy as f64 - centre;
                let dx = ix as f64 - centre;
                if (dz * dz + dy * dy + dx * dx).sqrt() < 3.0 {
                    sphere_vals[iz * N * N + iy * N + ix] = 100.0;
                }
            }
        }
    }

    let config = FrangiConfig {
        scales: vec![1.0, 2.0],
        alpha: 0.5,
        beta: 0.5,
        gamma: 15.0,
        polarity: VesselPolarity::Bright,
    };

    let tube_image = make_image(tube_vals, [N, N, N]);
    let sphere_image = make_image(sphere_vals, [N, N, N]);

    let filter = FrangiVesselnessFilter::new(config);
    let tube_out = filter.apply(&tube_image).unwrap();
    let sphere_out = filter.apply(&sphere_image).unwrap();

    let (tube_v, _) = extract_vec_infallible(&tube_out);
    let (sphere_v, _) = extract_vec_infallible(&sphere_out);
    // Centre index for 20×20×20 image.
    let c = 9usize;
    let tube_centre = tube_v[c * N * N + c * N + c];
    let sphere_centre = sphere_v[c * N * N + c * N + c];

    assert!(
        tube_centre > sphere_centre,
        "tube centre ({tube_centre:.4}) must exceed sphere centre ({sphere_centre:.4})"
    );
}

/// **Test 5 — Dark vessel polarity gate.**
///
/// With `bright_vessels = false`, a bright tube returns zero vesselness
/// everywhere (all eigenvalues are negative, gate requires positive).
#[test]
fn test_frangi_dark_vessel_gate_rejects_bright_tube() {
    const N: usize = 12;
    let mut vals = vec![0.0f32; N * N * N];
    let centre = (N as f64 - 1.0) / 2.0;
    for iz in 0..N {
        for iy in 0..N {
            for ix in 0..N {
                let dy = iy as f64 - centre;
                let dx = ix as f64 - centre;
                if (dy * dy + dx * dx).sqrt() < 2.5 {
                    vals[iz * N * N + iy * N + ix] = 100.0;
                }
            }
        }
    }

    let image = make_image(vals, [N, N, N]);
    let config = FrangiConfig {
        scales: vec![1.5],
        polarity: VesselPolarity::Dark, // dark-vessel mode
        ..Default::default()
    };
    let filter = FrangiVesselnessFilter::new(config);
    let out = filter.apply(&image).expect("frangi apply failed");

    let (v, _) = extract_vec_infallible(&out);

    let c = N / 2;
    let centre_val = v[c * N * N + c * N + c];
    assert!(
        centre_val < 1e-6,
        "dark-vessel mode must reject bright tube; centre V = {centre_val}"
    );
}
