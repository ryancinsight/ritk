use super::{
    BinaryMinMaxCurvatureFlowConfig, BinaryMinMaxCurvatureFlowImageFilter,
    MinMaxCurvatureFlowConfig, MinMaxCurvatureFlowImageFilter,
};
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec;

type B = coeus_core::SequentialBackend;

// â”€â”€ T-2: stencil_radius=0 guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// `stencil_radius = 0` is not a valid configuration: the directional
/// threshold functions divide by `r` (radius as f64), which would produce
/// Inf/NaN, and the sphere stencil degenerates to a single voxel, making
/// the gate meaningless.  The guard asserts `stencil_radius >= 1`.
#[test]
#[should_panic(expected = "stencil_radius must be >= 1")]
fn min_max_stencil_radius_zero_panics() {
    let img = ts::make_image::<f32, B, 3>(vec![1.0_f32; 16], [1, 4, 4]);
    let _out = MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig {
        num_iterations: 1,
        time_step: 0.05,
        stencil_radius: 0,
    })
    .apply(&img);
}

#[test]
fn binary_matches_sitk() {
    let (h, w) = (8usize, 8usize);
    let mut data = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let base = if (2..6).contains(&y) && (2..6).contains(&x) {
                1.0
            } else {
                -1.0
            };
            data[y * w + x] = base + 0.1 * ((x + y) % 3) as f32 - 0.1;
        }
    }
    let img = ts::make_image::<f32, B, 3>(data, [1, h, w]);
    let out = BinaryMinMaxCurvatureFlowImageFilter::new(BinaryMinMaxCurvatureFlowConfig {
        num_iterations: 3,
        time_step: 0.05,
        stencil_radius: 2,
        threshold: 0.0,
    })
    .apply(&img)
    .unwrap();
    let (got, _) = extract_vec(&out).unwrap();
    let expect: [f32; 64] = [
        -1.1, -1.0, -0.91031, -1.1, -1.0, -0.91027, -1.1, -1.0, -1.0, -0.92629, -1.1, -1.0,
        -0.91108, -1.1, -1.0, -0.90996, -0.91031, -1.1, 0.90863, 1.1, 0.90984, 0.9089, -0.90878,
        -1.1, -1.1, -1.0, 1.1, 0.90858, 1.0, 1.1, -1.1, -1.00515, -1.0, -0.91108, 0.90984, 1.0,
        1.1, 0.9092, -1.0, -0.91003, -0.91027, -1.1, 0.9089, 1.1, 0.9092, 0.90858, -0.91248, -1.1,
        -1.1, -1.0, -0.90878, -1.1, -1.0, -0.91248, -1.10998, -1.00526, -1.0, -0.90996, -1.1,
        -1.00515, -0.91003, -1.1, -1.00526, -0.90376,
    ];
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        assert!(
            (g - e).abs() < 2e-3,
            "voxel {i}: ritk {g} vs sitk {e} (diff {})",
            (g - e).abs()
        );
    }
}

#[test]
fn matches_sitk_min_max_curvature_flow() {
    let (h, w) = (8usize, 8usize);
    let mut data = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let base = 10.0 * ((x * 3 + y * 7) % 5) as f32;
            let blob = if (2..6).contains(&y) && (2..6).contains(&x) {
                50.0
            } else {
                0.0
            };
            data[y * w + x] = base + blob;
        }
    }
    let img = ts::make_image::<f32, B, 3>(data, [1, h, w]);
    let out = MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig {
        num_iterations: 3,
        time_step: 0.05,
        stencil_radius: 2,
    })
    .apply(&img)
    .unwrap();
    let (got, _) = extract_vec(&out).unwrap();

    let expect: [f32; 64] = [
        1.2649, 28.25748, 10.0, 38.25801, 20.93239, 0.0, 30.0, 11.19981, 20.0, 1.81966, 28.08779,
        10.0, 38.07228, 19.93572, 0.0, 30.0, 40.0, 21.01239, 49.96435, 80.0, 60.0, 86.3302,
        19.93572, 0.0, 10.0, 38.40163, 68.27138, 50.0, 80.0, 60.0, 38.07228, 20.93239, 28.39757,
        10.0, 88.23986, 70.0, 50.0, 80.0, 10.0, 38.25801, 0.0, 28.0715, 59.49664, 88.23986,
        68.27138, 49.96435, 28.08779, 10.0, 19.04184, 0.0, 28.0715, 10.0, 38.40163, 21.01239,
        1.81966, 28.25748, 39.24358, 19.04184, 0.0, 28.39757, 10.0, 40.0, 20.0, 1.2649,
    ];
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        assert!(
            (g - e).abs() < 2e-3,
            "voxel {i}: ritk {g} vs sitk {e} (diff {})",
            (g - e).abs()
        );
    }
}
