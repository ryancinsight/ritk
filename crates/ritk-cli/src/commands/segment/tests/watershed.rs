use super::*;

// ── Helper: uniform gradient and two-seed marker images ───────────────────────

/// Build a 3×3×3 image with uniform intensity 0.5.
///
/// Used as a synthetic gradient image for marker-watershed tests.
/// A flat gradient means all pairwise edge weights are equal, so the
/// watershed assigns labels purely by proximity to the seed voxels.
fn make_uniform_gradient_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let values = vec![0.5_f32; 27];
    let td = TensorData::new(values, Shape::new([3, 3, 3]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

/// Build a 3×3×3 marker image with two seeds at opposite corners.
///
/// Flat index 0  (z=0, y=0, x=0) → label 1.0
/// Flat index 26 (z=2, y=2, x=2) → label 2.0
/// All other voxels               → 0.0 (unmarked).
fn make_two_seed_marker_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let mut values = vec![0.0_f32; 27];
    values[0] = 1.0;
    values[26] = 2.0;
    let td = TensorData::new(values, Shape::new([3, 3, 3]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

// ── Positive: Watershed creates output with basin labels ──────────────────

#[test]
fn test_segment_watershed_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("labels.nii");

    ritk_io::write_nifti(&input, &make_ramp_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "watershed")).unwrap();

    assert!(output.exists(), "watershed output must be created");
    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(labels.shape(), [4, 4, 4], "shape must be preserved");

    labels.with_data_slice(|vals| {
        for &v in vals {
            assert!(v >= 0.0, "watershed labels must be non-negative, got {v}");
        }
    });
}

// ── Negative: marker-watershed missing markers path returns error ─────────

#[test]
fn test_segment_marker_watershed_missing_markers_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("image.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    let args = default_args(input.clone(), output.clone(), "marker-watershed");
    let result = run(args);
    assert!(
        result.is_err(),
        "marker-watershed without markers path must return error"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("marker"),
        "error must mention 'marker', got: {msg}"
    );
}

// ── Positive: Marker-watershed creates output with correct shape ──────────

/// Marker-watershed must produce a 3×3×3 output image when given valid
/// gradient and marker inputs.
///
/// Invariant: output shape == input shape == [3, 3, 3].
#[test]
fn test_segment_marker_watershed_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let gradient_path = dir.path().join("gradient.nii");
    let markers_path = dir.path().join("markers.nii");
    let output_path = dir.path().join("out.nii");

    ritk_io::write_nifti(&gradient_path, &make_uniform_gradient_image()).unwrap();
    ritk_io::write_nifti(&markers_path, &make_two_seed_marker_image()).unwrap();

    let args = SegmentArgs {
        markers: Some(markers_path.to_string_lossy().to_string()),
        ..default_args(
            gradient_path.clone(),
            output_path.clone(),
            "marker-watershed",
        )
    };
    let result = run(args);
    assert!(
        result.is_ok(),
        "marker-watershed must succeed with valid inputs: {:?}",
        result.err()
    );

    let img = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    assert_eq!(
        img.shape(),
        [3, 3, 3],
        "output shape must equal input shape [3, 3, 3]"
    );
}

/// Marker-watershed must propagate both seed labels into the output image.
///
/// With two seeds at opposite corners of a uniform-gradient 3×3×3 volume,
/// the flood fill assigns every voxel to the nearer seed.  At minimum the
/// seed voxels themselves carry their original labels, so label 1.0 and
/// label 2.0 must each appear at least once in the output.
#[test]
fn test_segment_marker_watershed_output_contains_both_basin_labels() {
    let dir = tempdir().unwrap();
    let gradient_path = dir.path().join("gradient.nii");
    let markers_path = dir.path().join("markers.nii");
    let output_path = dir.path().join("out.nii");

    ritk_io::write_nifti(&gradient_path, &make_uniform_gradient_image()).unwrap();
    ritk_io::write_nifti(&markers_path, &make_two_seed_marker_image()).unwrap();

    let args = SegmentArgs {
        markers: Some(markers_path.to_string_lossy().to_string()),
        ..default_args(
            gradient_path.clone(),
            output_path.clone(),
            "marker-watershed",
        )
    };
    run(args).expect("marker-watershed must succeed with valid inputs");

    let img = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
    img.with_data_slice(|vals| {
        let has_label_1 = vals.iter().any(|&v| v == 1.0_f32);
        let has_label_2 = vals.iter().any(|&v| v == 2.0_f32);
        assert!(
            has_label_1,
            "output must contain at least one voxel with label 1.0 (seed basin 1); \
             got values: {:?}",
            &vals[..vals.len().min(27)]
        );
        assert!(
            has_label_2,
            "output must contain at least one voxel with label 2.0 (seed basin 2); \
             got values: {:?}",
            &vals[..vals.len().min(27)]
        );
    });
}
