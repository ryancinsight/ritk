//! Tests for Sato vesselness filter.

use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

// Re-import using the crate's own paths (within ritk-core).
use ritk_image::Image as CoreImage;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, dims: [usize; 3]) -> CoreImage<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// Build a 3-D volume with a bright cylinder of radius `r` centred at
/// (cy, cx) running along the full z-axis.
fn make_tube(nz: usize, ny: usize, nx: usize, cy: f32, cx: f32, r: f32) -> Vec<f32> {
    (0..nz * ny * nx)
        .map(|fi| {
            let ix = (fi % nx) as f32;
            let iy = ((fi / nx) % ny) as f32;
            let dist = ((ix - cx).powi(2) + (iy - cy).powi(2)).sqrt();
            if dist <= r {
                1.0_f32
            } else {
                0.0_f32
            }
        })
        .collect()
}

/// Build a bright sphere centred at (cz, cy, cx) with radius `r`.
fn make_sphere(nz: usize, ny: usize, nx: usize, cz: f32, cy: f32, cx: f32, r: f32) -> Vec<f32> {
    (0..nz * ny * nx)
        .map(|fi| {
            let ix = (fi % nx) as f32;
            let iy = ((fi / nx) % ny) as f32;
            let iz = (fi / (ny * nx)) as f32;
            let dist = ((ix - cx).powi(2) + (iy - cy).powi(2) + (iz - cz).powi(2)).sqrt();
            if dist <= r {
                1.0_f32
            } else {
                0.0_f32
            }
        })
        .collect()
}

// ── Test 1 ────────────────────────────────────────────────────────────────

/// A bright cylinder along the z-axis must produce a high Sato response
/// at its centre compared to the background.
#[test]
fn test_cylindrical_tube_detects_line() {
    const N: usize = 32;
    let cy = 16.0_f32;
    let cx = 16.0_f32;
    let r = 3.0_f32;

    let data = make_tube(N, N, N, cy, cx, r);
    let image = make_image(data, [N, N, N]);

    let config = SatoConfig {
        scales: vec![1.5, 3.0],
        alpha: 0.5,
        polarity: VesselPolarity::Bright,
    };
    let filter = SatoLineFilter::new(config);
    let result = filter.apply(&image).expect("apply failed");

    let _device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let out: Vec<f32> = result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    // Centre column: z = any, y = 16, x = 16 (flat index = z*N*N + 16*N + 16).
    let mut centre_responses: Vec<f32> = (0..N).map(|iz| out[iz * N * N + 16 * N + 16]).collect();
    centre_responses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_centre = centre_responses[N / 2];

    // Far background voxels (corner strip x=0..2, y=0..2).
    let background: Vec<f32> = (0..N).map(|iz| out[iz * N * N]).collect();
    let mut bg = background.clone();
    bg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_bg = bg[N / 2];

    assert!(
        median_centre > 0.1,
        "centre-line median Sato response should be > 0.1, got {median_centre:.6}"
    );
    assert!(
        median_centre > median_bg * 3.0,
        "centre response ({median_centre:.6}) should exceed background ({median_bg:.6}) by 3×"
    );
}

// ── Test 2 ────────────────────────────────────────────────────────────────

/// A bright sphere has a lower peak Sato response than the cylinder tube
/// produced by test 1, because spheres are blob-like, not line-like.
#[test]
fn test_sphere_lower_response_than_tube() {
    const N: usize = 32;

    let config = SatoConfig {
        scales: vec![1.5, 3.0],
        alpha: 0.5,
        polarity: VesselPolarity::Bright,
    };
    let tube_data = make_tube(N, N, N, 16.0, 16.0, 3.0);
    let sphere_data = make_sphere(N, N, N, 16.0, 16.0, 16.0, 4.0);
    let filter = SatoLineFilter::new(config);

    let tube_img = make_image(tube_data, [N, N, N]);
    let sphere_img = make_image(sphere_data, [N, N, N]);

    let tube_out: Vec<f32> = filter
        .apply(&tube_img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let sphere_out: Vec<f32> = filter
        .apply(&sphere_img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let tube_peak = tube_out.iter().cloned().fold(0.0_f32, f32::max);
    let sphere_peak = sphere_out.iter().cloned().fold(0.0_f32, f32::max);

    assert!(
        tube_peak > sphere_peak,
        "tube peak ({tube_peak:.6}) should exceed sphere peak ({sphere_peak:.6})"
    );
}

// ── Test 3 ────────────────────────────────────────────────────────────────

/// With `bright_tubes = true`, a dark tube (intensity 0, background 1)
/// must produce near-zero response everywhere.
#[test]
fn test_dark_tube_rejected_by_bright_gate() {
    const N: usize = 24;

    // Invert: background = 1, tube = 0.
    let tube_mask = make_tube(N, N, N, 12.0, 12.0, 2.5);
    let dark_tube: Vec<f32> = tube_mask.iter().map(|&v| 1.0 - v).collect();

    let config = SatoConfig {
        scales: vec![1.5],
        alpha: 0.5,
        polarity: VesselPolarity::Bright, // bright gate — should reject the dark tube
    };
    let filter = SatoLineFilter::new(config);
    let result = filter.apply(&make_image(dark_tube, [N, N, N])).unwrap();
    let out: Vec<f32> = result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let max_resp = out.iter().cloned().fold(0.0_f32, f32::max);
    assert!(
        max_resp < 0.05,
        "bright-gate must reject dark tube; max response = {max_resp:.6}"
    );
}

// ── Test 4 ────────────────────────────────────────────────────────────────

/// A uniform image has zero Hessian everywhere, so all Sato responses are zero.
#[test]
fn test_uniform_image_zero_response() {
    const N: usize = 16;
    let data = vec![0.5_f32; N * N * N];

    let config = SatoConfig::default();
    let filter = SatoLineFilter::new(config);
    let result = filter.apply(&make_image(data, [N, N, N])).unwrap();
    let out: Vec<f32> = result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let max_resp = out.iter().cloned().fold(0.0_f32, f32::max);
    assert!(
        max_resp < 1e-5,
        "uniform image must give zero response; max = {max_resp:.2e}"
    );
}

// ── Test 5 ────────────────────────────────────────────────────────────────

/// All output voxels must be finite for any non-trivial input.
#[test]
fn test_response_all_finite() {
    const N: usize = 20;

    // Sinusoidal phantom: exercises all code paths including the perp term.
    let data: Vec<f32> = (0..N * N * N)
        .map(|fi| {
            let ix = fi % N;
            let iy = (fi / N) % N;
            let iz = fi / (N * N);
            (std::f32::consts::PI * ix as f32 / N as f32).sin()
                * (std::f32::consts::PI * iy as f32 / N as f32).cos()
                * (0.5 + iz as f32 / N as f32)
        })
        .collect();

    let config = SatoConfig {
        scales: vec![1.0, 2.0],
        alpha: 1.0,
        polarity: VesselPolarity::Bright,
    };
    let filter = SatoLineFilter::new(config);
    let result = filter.apply(&make_image(data, [N, N, N])).unwrap();
    let out: Vec<f32> = result
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "voxel {i} is non-finite: {v}");
    }
}
