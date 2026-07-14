use super::*;

#[test]
fn native_region_growing_cli_family_matches_legacy_boundaries_exactly() {
    use ritk_segmentation::{
        ConfidenceConnectedFilter, ConnectedThresholdFilter, NeighborhoodConnectedFilter,
    };

    let dir = tempdir().unwrap();
    for method in [
        SegmentMethod::ConnectedThreshold,
        SegmentMethod::ConfidenceConnected,
        SegmentMethod::NeighborhoodConnected,
    ] {
        let label = format!("{method:?}");
        let input = dir.path().join(format!("{label}-input.nii"));
        let output = dir.path().join(format!("{label}-output.nii"));
        let fixture = make_sphere_image();
        let mut args = default_args(input.clone(), output.clone(), method);
        args.lower = Some(100.0);
        args.upper = Some(255.0);
        args.seed = Some("2,2,2".to_string());
        let expected = match &args.method {
            SegmentMethod::ConnectedThreshold => {
                ConnectedThresholdFilter::new([2, 2, 2], 100.0, 255.0).apply(&fixture)
            }
            SegmentMethod::ConfidenceConnected => {
                ConfidenceConnectedFilter::new([2, 2, 2], 100.0, 255.0)
                    .with_multiplier(args.multiplier)
                    .expect("test multiplier is valid")
                    .with_max_iterations(args.max_iterations)
                    .apply(&fixture)
            }
            SegmentMethod::NeighborhoodConnected => {
                let radius = args.neighborhood_radius;
                NeighborhoodConnectedFilter::new([2, 2, 2], 100.0, 255.0)
                    .with_radius([radius; 3])
                    .apply(&fixture)
            }
            _ => unreachable!("test enumerates only region-growing methods"),
        };
        ritk_io::write_nifti(&input, &fixture).unwrap();

        run(args).unwrap();
        let actual = crate::commands::read_image_native(&output)
            .expect("region-growing output is natively readable");

        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(*actual.origin(), *expected.origin());
        assert_eq!(*actual.spacing(), *expected.spacing());
        assert_eq!(*actual.direction(), *expected.direction());
        assert_eq!(
            actual.data_slice().expect("contiguous native mask"),
            expected.data_slice().as_ref(),
            "{label} native CLI output diverged from its legacy public boundary"
        );
    }
}

#[test]
fn region_growing_rejects_nan_bounds_before_io() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("output.nii");
    let mut args = default_args(
        dir.path().join("missing.nii"),
        output.clone(),
        SegmentMethod::ConnectedThreshold,
    );
    args.lower = Some(f32::NAN);
    args.upper = Some(1.0);
    args.seed = Some("0,0,0".to_string());
    let error = run(args).expect_err("NaN bounds must be rejected before I/O");
    assert!(error.to_string().contains("must not be NaN"));
    assert!(!output.exists());
}

#[test]
fn confidence_connected_rejects_invalid_multiplier_before_io() {
    for multiplier in [f32::NAN, f32::INFINITY, -1.0] {
        let mut args = default_args(
            "missing.nii".into(),
            "output.nii".into(),
            SegmentMethod::ConfidenceConnected,
        );
        args.multiplier = multiplier;
        let error = run(args).expect_err("invalid multiplier must be rejected before I/O");
        assert!(error.to_string().contains("finite and non-negative"));
    }
}

#[test]
fn region_growing_rejects_nonnative_output_before_io() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("output.png");
    let mut args = default_args(
        dir.path().join("input.vtk"),
        output.clone(),
        SegmentMethod::NeighborhoodConnected,
    );
    args.lower = Some(0.0);
    args.upper = Some(1.0);
    args.seed = Some("0,0,0".to_string());
    let error = run(args).expect_err("nonnative output must be rejected before I/O");
    assert_eq!(
        error.to_string(),
        "neighborhood-connected requires native input/output formats"
    );
    assert!(!output.exists());
}

// ── Positive: Connected-threshold grows sphere region ─────────────────────

/// Seeding at the centre of the sphere must grow exactly the 7 high-intensity
/// voxels (centre + 6 face-adjacent neighbours).
#[test]
fn test_segment_connected_threshold_grows_sphere_from_centre_seed() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");

    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    run(SegmentArgs {
        input: input.clone(),
        output: output.clone(),
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    })
    .unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 7,
        "connected-threshold from centre seed must grow exactly 7 sphere voxels"
    );
}

// ── Positive: Connected-threshold output is strictly binary ───────────────

#[test]
fn test_segment_connected_threshold_output_is_strictly_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.mha");
    let output = dir.path().join("mask.mha");

    ritk_io::write_metaimage(&input, &make_sphere_image()).unwrap();

    run(SegmentArgs {
        input: input.clone(),
        output: output.clone(),
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    })
    .unwrap();

    let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "connected-threshold output must be strictly binary, got {v}"
            );
        }
    });
}

// ── Negative: connected-threshold missing --lower ─────────────────────────

#[test]
fn test_segment_connected_threshold_missing_lower_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: None,
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --lower must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--lower is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold missing --upper ─────────────────────────

#[test]
fn test_segment_connected_threshold_missing_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: None,
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --upper must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--upper is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold missing --seed ──────────────────────────

#[test]
fn test_segment_connected_threshold_missing_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: None,
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--seed is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold lower > upper ───────────────────────────

#[test]
fn test_segment_connected_threshold_lower_gt_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(255.0),
        upper: Some(100.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "lower > upper must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("must be \u{2264}") || msg.contains("must be <=") || msg.contains('\u{2264}'),
        "error must explain the bound constraint, got: {msg}"
    );
}

// ── Negative: out-of-bounds seed returns error ────────────────────────────

#[test]
fn test_segment_connected_threshold_out_of_bounds_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("99,99,99".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "out-of-bounds seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("out of bounds"),
        "error must explain the bounds problem, got: {msg}"
    );
}

// ── Negative: malformed seed string returns error ─────────────────────────

#[test]
fn test_segment_malformed_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::ConnectedThreshold,
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "malformed seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Z,Y,X"),
        "error must explain the expected format, got: {msg}"
    );
}

// -- confidence-connected: positive ---------------------------------------

#[test]
fn test_segment_confidence_connected_grows_region() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::ConfidenceConnected,
    );
    args.lower = Some(150.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [5, 5, 5], "shape must match input");
    let n_fg = count_foreground(&mask);
    assert!(n_fg > 0, "must find at least one foreground voxel, got 0");
}

#[test]
fn test_segment_confidence_connected_output_is_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::ConfidenceConnected,
    );
    args.lower = Some(150.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    mask.with_data_slice(|vals| {
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "all voxels must be 0.0 or 1.0, found {v}"
            );
        }
    });
}

#[test]
fn test_segment_confidence_connected_missing_lower_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, SegmentMethod::ConfidenceConnected);
    args.upper = Some(1.5);
    args.seed = Some("2,2,2".to_string());
    let result = run(args);
    assert!(result.is_err(), "--lower missing must produce an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--lower"),
        "error must mention --lower, got: {msg}"
    );
}

#[test]
fn test_segment_confidence_connected_missing_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, SegmentMethod::ConfidenceConnected);
    args.lower = Some(0.5);
    args.seed = Some("2,2,2".to_string());
    let result = run(args);
    assert!(result.is_err(), "--upper missing must produce an error");
}

#[test]
fn test_segment_confidence_connected_missing_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, SegmentMethod::ConfidenceConnected);
    args.lower = Some(0.5);
    args.upper = Some(1.5);
    let result = run(args);
    assert!(result.is_err(), "--seed missing must produce an error");
}

// -- neighborhood-connected: positive -------------------------------------

#[test]
fn test_segment_neighborhood_connected_grows_region() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::NeighborhoodConnected,
    );
    args.lower = Some(0.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [5, 5, 5]);
    let n_fg = count_foreground(&mask);
    assert!(n_fg > 0, "must find foreground voxels, got 0");
}

#[test]
fn test_segment_neighborhood_connected_output_is_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::NeighborhoodConnected,
    );
    args.lower = Some(0.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    mask.with_data_slice(|vals| {
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "all voxels must be 0.0 or 1.0, found {v}"
            );
        }
    });
}
