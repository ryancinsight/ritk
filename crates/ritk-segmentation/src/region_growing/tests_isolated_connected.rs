use super::{IsolatedConnectedConfig, IsolatedConnectedFilter, IsolationThreshold};
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

fn two_blob_values(bridge: f32) -> Vec<f32> {
    let (ny, nx) = (10usize, 16);
    let mut values = vec![0.0; ny * nx];
    for y in 3..=6 {
        for x in 1..=4 {
            values[y * nx + x] = 100.0;
        }
        for x in 11..=14 {
            values[y * nx + x] = 100.0;
        }
    }
    for y in 4..=5 {
        for x in 5..=10 {
            values[y * nx + x] = bridge;
        }
    }
    values
}

/// Two intensity-100 blobs joined by an intensity-150 bridge. With band floor
/// 50 and ceiling 200, the separating upper threshold lands just below 150, so
/// the region grown from seed1 keeps its blob (100 ≤ 149) but excludes the
/// bridge (150) and therefore the second blob/seed.
#[test]
fn isolated_connected_separates_two_blobs() {
    let (ny, nx) = (10usize, 16);
    let img = make(two_blob_values(150.0), [1, ny, nx]);
    let config =
        IsolatedConnectedConfig::new(50.0, 200.0, 1.0, 1.0, IsolationThreshold::Upper).expect("infallible: validated precondition");
    let f = IsolatedConnectedFilter::new([0, 4, 2], [0, 4, 13], config);
    let result = f.apply(&img).expect("infallible: validated precondition");
    assert!(!result.thresholding_failed());
    let out = result.into_image();
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov[4 * nx + 2], 1.0, "seed1's blob must be kept");
    assert_eq!(ov[4 * nx + 13], 0.0, "seed2's blob must be isolated out");
    assert_eq!(ov[4 * nx + 7], 0.0, "the bridge must be excluded");
    // Every kept voxel is part of blob 1 (columns 1..=4).
    for y in 0..ny {
        for x in 0..nx {
            if ov[y * nx + x] != 0.0 {
                assert!(
                    (1..=4).contains(&x),
                    "kept voxel outside blob 1 at ({y},{x})"
                );
            }
        }
    }
}

/// Custom replace value is written to the kept region.
#[test]
fn isolated_connected_custom_replace_value() {
    let (ny, nx) = (6usize, 6);
    let mut v = vec![0.0f32; ny * nx];
    for y in 1..=4 {
        for x in 1..=4 {
            v[y * nx + x] = 100.0;
        }
    }
    let img = make(v, [1, ny, nx]);
    let config =
        IsolatedConnectedConfig::new(50.0, 200.0, 7.0, 1.0, IsolationThreshold::Upper).expect("infallible: validated precondition");
    let f = IsolatedConnectedFilter::new([0, 2, 2], [0, 0, 0], config);
    let result = f.apply(&img).expect("infallible: validated precondition");
    assert!(!result.thresholding_failed());
    let out = result.into_image();
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov[2 * nx + 2], 7.0, "kept region uses the replace value");
    assert!(ov.iter().all(|&v| v == 0.0 || v == 7.0));
}

#[test]
fn lower_threshold_search_separates_low_bridge() {
    let config =
        IsolatedConnectedConfig::new(0.0, 200.0, 1.0, 1.0, IsolationThreshold::Lower).expect("infallible: validated precondition");
    let filter = IsolatedConnectedFilter::new([0, 4, 2], [0, 4, 13], config);
    let result = filter
        .apply(&make(two_blob_values(40.0), [1, 10, 16]))
        .expect("infallible: validated precondition");
    assert!(!result.thresholding_failed());
    let output = result.into_image();
    let (values, _) = extract_vec_infallible(&output);
    assert_eq!(values.iter().filter(|&&value| value == 1.0).count(), 16);
    assert_eq!(values[4 * 16 + 2], 1.0);
    assert_eq!(values[4 * 16 + 13], 0.0);
    assert_eq!(values[4 * 16 + 7], 0.0);
}

#[test]
fn native_and_legacy_outputs_are_exact_with_nonidentity_geometry() {
    let values = two_blob_values(150.0);
    let legacy = make(values.clone(), [1, 10, 16]);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let native = NativeImage::from_flat_on(
        values,
        [1, 10, 16],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("infallible: validated precondition");
    let config =
        IsolatedConnectedConfig::new(50.0, 200.0, 1.0, 1.0, IsolationThreshold::Upper).expect("infallible: validated precondition");
    let filter = IsolatedConnectedFilter::new([0, 4, 2], [0, 4, 13], config);
    let expected = filter.apply(&legacy).expect("infallible: validated precondition");
    let actual = filter.apply_native(&native, &SequentialBackend).expect("infallible: validated precondition");
    assert_eq!(actual.thresholding_failed(), expected.thresholding_failed());
    let expected = expected.into_image();
    let actual = actual.into_image();
    assert_eq!(
        actual.data_slice().expect("infallible: validated precondition"),
        expected
            .data_slice()
            .expect("invariant: contiguous host storage")
    );
    assert_eq!(*actual.origin(), origin);
    assert_eq!(*actual.spacing(), spacing);
    assert_eq!(*actual.direction(), direction);
}

#[test]
fn lower_threshold_failure_retains_the_final_mask() {
    let config =
        IsolatedConnectedConfig::new(50.0, 200.0, 1.0, 1.0, IsolationThreshold::Lower).expect("infallible: validated precondition");
    let result = IsolatedConnectedFilter::new([0, 4, 2], [0, 4, 13], config)
        .apply(&make(two_blob_values(150.0), [1, 10, 16]))
        .expect("infallible: validated precondition");
    assert!(result.thresholding_failed());
    let (values, _) = extract_vec_infallible(&result.into_image());
    assert_eq!(values.iter().filter(|&&value| value == 1.0).count(), 0);
    assert_eq!(values[4 * 16 + 2], 0.0);
    assert_eq!(values[4 * 16 + 13], 0.0);
}

#[test]
fn configuration_and_input_failures_are_exact() {
    assert!(IsolatedConnectedConfig::new(0.0, 1.0, 0.0, 1.0, IsolationThreshold::Upper).is_ok());
    assert_eq!(
        IsolatedConnectedConfig::new(f32::NAN, 1.0, 1.0, 1.0, IsolationThreshold::Upper)
            .unwrap_err()
            .to_string(),
        "isolated connected bounds must be finite and ordered, got [NaN, 1]"
    );
    assert_eq!(
        IsolatedConnectedConfig::new(0.0, 1.0, f32::INFINITY, 1.0, IsolationThreshold::Upper)
            .unwrap_err()
            .to_string(),
        "isolated connected replacement must be finite, got inf"
    );
    assert_eq!(
        IsolatedConnectedConfig::new(0.0, 1.0, 1.0, 0.0, IsolationThreshold::Upper)
            .unwrap_err()
            .to_string(),
        "isolated connected tolerance must be finite and positive, got 0"
    );
    let config =
        IsolatedConnectedConfig::new(0.0, 2.0, 1.0, 0.1, IsolationThreshold::Upper).expect("infallible: validated precondition");
    let invalid_seed = IsolatedConnectedFilter::new([0, 0, 0], [0, 0, 2], config);
    assert_eq!(
        invalid_seed
            .apply(&make(vec![0.0, 1.0], [1, 1, 2]))
            .unwrap_err()
            .to_string(),
        "isolated connected seed2 [0, 0, 2] is outside shape [1, 1, 2]"
    );
    let nonfinite = IsolatedConnectedFilter::new([0, 0, 0], [0, 0, 1], config);
    assert_eq!(
        nonfinite
            .apply(&make(vec![0.0, f32::NAN], [1, 1, 2]))
            .unwrap_err()
            .to_string(),
        "isolated connected sample at flat index 1 must be finite, got NaN"
    );
}
